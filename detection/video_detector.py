import cv2
import numpy as np
import base64
import math
import time
from collections import defaultdict

try:
    from ultralytics import YOLO
    yolo_model    = YOLO('yolov8n.pt')
    YOLO_AVAILABLE = True
    print('[YOLO] Model loaded')
except Exception as e:
    YOLO_AVAILABLE = False
    print(f'[YOLO] Not available: {e}')

# Chain snatching pose model
try:
    from detection.chain_snatch import get_chain_detector
    chain_detector = get_chain_detector(fps=10)
    CHAIN_AVAILABLE = True
except Exception as e:
    CHAIN_AVAILABLE = False
    print(f'[CHAIN] Not available: {e}')

WEAPON_CLASSES = {
    'knife', 'gun', 'pistol', 'rifle',
    'scissors', 'baseball bat', 'bottle'
}

person_history = defaultdict(list)
threat_scores  = defaultdict(float)
prev_gray      = None
HISTORY_WINDOW = 120
SCORE_WEIGHTS  = {
    'LOITERING':         30,
    'PERSON_FOLLOWING':  45,
    'STALKING_PATTERN':  55,
    'ABNORMAL_MOVEMENT': 25,
    'SNATCHING':         65,
}

SNATCH_CLASSES = {'backpack', 'handbag', 'suitcase', 'cell phone'}
VEHICLE_CLASSES = {'motorcycle', 'bicycle'}
SUSPICIOUS_CLASSES = SNATCH_CLASSES | VEHICLE_CLASSES


def decode_frame(b64_data: str):
    try:
        if ',' in b64_data:
            b64_data = b64_data.split(',')[1]
        img_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f'[DECODE] Error: {e}')
        return None


def analyse_frame(b64_data: str) -> dict:
    frame = decode_frame(b64_data)
    if frame is None:
        return {'detections': [], 'tracked_persons': [],
                'predictive_threats': [], 'threat_scores': {}}

    detections      = []
    tracked_persons = []

    if YOLO_AVAILABLE:
        try:
            results = yolo_model.track(
                frame,
                persist=True,
                tracker='bytetrack.yaml',
                conf=0.40,
                verbose=False,
            )
            if results and results[0].boxes:
                boxes = results[0].boxes
                names = yolo_model.names
                for i, box in enumerate(boxes):
                    label = names[int(box.cls[0])]
                    conf  = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    det = {'label': label, 'confidence': round(conf, 3),
                           'bbox': [x1, y1, x2, y2], 'cx': cx, 'cy': cy}
                    
                    if boxes.id is not None:
                        tid = int(boxes.id[i])
                        det['track_id'] = tid
                        
                        if label == 'person':
                            tracked_persons.append({
                                'id': tid, 'cx': cx, 'cy': cy,
                                'bbox': [x1, y1, x2, y2]
                            })
                        elif label in SUSPICIOUS_CLASSES:
                            # Also track suspicious objects for snatching logic
                            tracked_persons.append({
                                'id': tid, 'cx': cx, 'cy': cy,
                                'bbox': [x1, y1, x2, y2],
                                'label': label
                            })
                    detections.append(det)
        except Exception as e:
            print(f'[YOLO] Inference error: {e}')

    _update_history(tracked_persons)
    ids = [p['id'] for p in tracked_persons]
    predictive_threats = []

    for p in tracked_persons:
        pid    = p['id']
        result = (_check_loitering(pid) or
                  _check_following(ids) or
                  _check_stalking(ids) or
                  _check_snatching(pid) or
                  _check_abnormal_movement(frame, tracked_persons))
        _update_threat_score(pid, result['type'] if result else None)
        evt = _classify_score(threat_scores[pid])
        if evt:
            evt['person_id'] = pid
            predictive_threats.append(evt)

    # ── Chain snatching model ────────────────────────────────────────────
    chain_alert = None
    if CHAIN_AVAILABLE:
        try:
            threat_event = chain_detector.process_frame(frame)
            if threat_event:
                chain_alert = threat_event.to_sentinel_payload()
                print(f'[CHAIN] Snatching detected! conf={threat_event.confidence:.0%} sev={threat_event.severity}')
        except Exception as e:
            print(f'[CHAIN] Error: {e}')

    return {
        'detections':         detections,
        'tracked_persons':    tracked_persons,
        'predictive_threats': predictive_threats,
        'threat_scores':      {str(k): round(v,1)
                              for k,v in threat_scores.items()},
        'chain_alert':        chain_alert,
    }


def classify_reactive_threat(detections: list) -> dict | None:
    for det in detections:
        label = det['label'].lower()
        conf  = det['confidence']
        if label in WEAPON_CLASSES and conf >= 0.40:
            level = 'HIGH' if label in {'knife','gun','pistol','rifle'} else 'MEDIUM'
            return {'type': 'WEAPON_DETECTED', 'level': level,
                    'confidence': conf, 'detail': label,
                    'bbox': det.get('bbox')}
    return None


def _update_history(tracked_persons):
    now = time.time()
    for p in tracked_persons:
        person_history[p['id']].append((p['cx'], p['cy'], now))
    for pid in list(person_history.keys()):
        person_history[pid] = [
            e for e in person_history[pid]
            if now - e[2] < HISTORY_WINDOW
        ]
        if not person_history[pid]:
            del person_history[pid]


def _check_loitering(pid):
    h = person_history.get(pid, [])
    if len(h) < 10: return None
    now    = time.time()
    recent = [(cx,cy) for cx,cy,ts in h if now-ts < 90]
    if len(recent) < 10: return None
    xs = [p[0] for p in recent]; ys = [p[1] for p in recent]
    mx = sum(xs)/len(xs); my = sum(ys)/len(ys)
    sx = (sum((x-mx)**2 for x in xs)/len(xs))**0.5
    sy = (sum((y-my)**2 for y in ys)/len(ys))**0.5
    spread = (sx**2+sy**2)**0.5
    dwell  = now - h[0][2]
    if spread < 40 and dwell > 90:
        return {'type':'LOITERING','level':'MEDIUM',
                'confidence':0.78,'dwell_seconds':round(dwell)}
    return None


def _cosine_sim(v1, v2):
    m1=math.hypot(*v1); m2=math.hypot(*v2)
    if m1==0 or m2==0: return 0.0
    return (v1[0]*v2[0]+v1[1]*v2[1])/(m1*m2)

def _mvec(pts):
    if len(pts)<4: return (0.0,0.0)
    return (pts[-1][0]-pts[0][0], pts[-1][1]-pts[0][1])


def _check_following(ids):
    now = time.time()
    for i in ids:
        for j in ids:
            if i==j: continue
            h1=[(cx,cy) for cx,cy,ts in person_history.get(i,[]) if now-ts<15]
            h2=[(cx,cy) for cx,cy,ts in person_history.get(j,[]) if now-ts<15]
            if len(h1)<8 or len(h2)<8: continue
            v1=_mvec(h1[-8:]); v2=_mvec(h2[-8:])
            align=_cosine_sim(v1,v2)
            dist=math.hypot(h1[-1][0]-h2[-1][0],h1[-1][1]-h2[-1][1])
            sp1=math.hypot(*v1); sp2=math.hypot(*v2)
            if align>0.92 and 60<dist<220 and abs(sp1-sp2)<15 and sp1>5:
                return {'type':'PERSON_FOLLOWING','level':'HIGH',
                        'confidence':round(align,2),
                        'follower_id':j,'target_id':i}
    return None


def _check_stalking(ids):
    now = time.time()
    for i in ids:
        for j in ids:
            if i==j: continue
            h1=[(cx,cy,ts) for cx,cy,ts in person_history.get(i,[]) if now-ts<300]
            h2=[(cx,cy,ts) for cx,cy,ts in person_history.get(j,[]) if now-ts<300]
            if len(h1)<30 or len(h2)<30: continue
            prox=sum(1 for (x1,y1,t1) in h1 for (x2,y2,t2) in h2
                     if abs(t1-t2)<2 and math.hypot(x1-x2,y1-y2)<180)
            ratio=prox/len(h1)
            if ratio>0.55 and len(h1)>50:
                return {'type':'STALKING_PATTERN','level':'HIGH',
                        'confidence':round(ratio,2),
                        'suspect_id':j,'target_id':i}
    return None


def _check_abnormal_movement(frame, tracked_persons):
    global prev_gray
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None or prev_gray.shape != gray.shape:
            prev_gray = gray; return None
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        prev_gray = gray
        pflows = []
        for p in tracked_persons:
            x1,y1,x2,y2 = p['bbox']
            region = flow[max(0,y1):y2, max(0,x1):x2]
            if region.size == 0: continue
            pflows.append({'id':p['id'],
                           'dx':float(np.mean(region[:,:,0])),
                           'dy':float(np.mean(region[:,:,1]))})
        if len(pflows)<2: return None
        mdx=sum(f['dx'] for f in pflows)/len(pflows)
        mdy=sum(f['dy'] for f in pflows)/len(pflows)
        for f in pflows:
            dev=math.hypot(f['dx']-mdx,f['dy']-mdy)
            if dev>12:
                return {'type':'ABNORMAL_MOVEMENT','level':'MEDIUM',
                        'confidence':min(round(dev/20,2),0.99),
                        'person_id':f['id']}
    except Exception as e:
        print(f'[FLOW] Error: {e}')
    return None


def _check_snatching(pid):
    """
    Detect Snatching based on high velocity and proximity changes.
    """
    h = person_history.get(pid, [])
    if len(h) < 5: return None
    
    # Calculate current velocity (pixels per frame roughly)
    v = _mvec(h[-5:])
    speed = math.hypot(*v)
    
    # If moving very fast (speed > 25)
    if speed > 25:
        return {
            'type': 'SNATCHING',
            'level': 'HIGH',
            'confidence': min(round(speed/40, 2), 0.95),
            'detail': 'High speed movement detected'
        }
    return None


def _update_threat_score(pid, triggered_type):
    if triggered_type and triggered_type in SCORE_WEIGHTS:
        threat_scores[pid] = min(
            threat_scores[pid] + SCORE_WEIGHTS[triggered_type], 100)
    else:
        threat_scores[pid] = max(threat_scores[pid] - 1.5, 0)


def _classify_score(score):
    if score >= 90:
        return {'type':'PREDICTED_THREAT','level':'HIGH','score':round(score,1)}
    if score >= 70:
        return {'type':'BEHAVIOUR_WARNING','level':'MEDIUM','score':round(score,1)}
    return None
