import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.ext.asyncio import AsyncSession
import json, time, asyncio, base64, datetime

from database.models   import init_db, get_db
from database.crud     import save_incident, get_incidents, update_user_gps
from auth.routes       import router as auth_router
from websocket_manager import manager
from detection.video_detector import analyse_frame, classify_reactive_threat
from detection.audio_detector import analyse_audio
from report.generator  import generate_report
from utils             import haversine, format_distance

app = FastAPI(title='Sentinel AI', version='1.0.0')

app.add_middleware(CORSMiddleware,
    allow_origins=['*'], allow_credentials=True,
    allow_methods=['*'], allow_headers=['*'])

app.include_router(auth_router, prefix='/auth', tags=['Auth'])


@app.on_event('startup')
async def startup():
    await init_db()
    print('[STARTUP] Sentinel AI backend running')


# In-memory GPS store
# room_id -> { 'cctv': (lat, lon), 'police': (lat, lon) }
gps_store:     dict = {}
alert_cooldown: dict = {}
COOLDOWN_SECONDS = 8


def is_on_cooldown(room_id: str, threat_type: str) -> bool:
    now = time.time()
    last = alert_cooldown.get(room_id, {}).get(threat_type, 0)
    return (now - last) < COOLDOWN_SECONDS

def set_cooldown(room_id: str, threat_type: str):
    alert_cooldown.setdefault(room_id, {})[threat_type] = time.time()


async def dispatch_alert(room_id: str, threat: dict,
                         db: AsyncSession, detection_ts: int = None,
                         frame_data: str = None):
    if detection_ts is None:
        detection_ts = int(time.time() * 1000)

    coords        = gps_store.get(room_id, {})
    cctv_coords   = coords.get('cctv',   (0.0, 0.0))
    police_coords = coords.get('police', (0.0, 0.0))
    dist_m        = haversine(*cctv_coords, *police_coords)
    threat['timestamp'] = datetime.datetime.utcnow().isoformat()

    # Phase 1: immediate alert, no report yet
    await manager.alert_police(room_id, {
        'type':         'crime_alert',
        'status':       'alerting',
        'threat':        threat,
        'distance_m':    dist_m,
        'distance_fmt':  format_distance(dist_m),
        'cctv_lat':      cctv_coords[0],
        'cctv_lon':      cctv_coords[1],
        'report':        None,
        'maps_url': f'https://maps.google.com/?q={cctv_coords[0]},{cctv_coords[1]}',
        'detection_ts':  detection_ts,
        'frame_data':    frame_data,
    })

    # Phase 2: generate report then push update
    report = await generate_report(threat, dist_m)
    await manager.alert_police(room_id, {
        'type':         'crime_alert',
        'status':       'report_ready',
        'threat':        threat,
        'distance_m':    dist_m,
        'distance_fmt':  format_distance(dist_m),
        'cctv_lat':      cctv_coords[0],
        'cctv_lon':      cctv_coords[1],
        'report':        report,
        'maps_url': f'https://maps.google.com/?q={cctv_coords[0]},{cctv_coords[1]}',
        'detection_ts':  detection_ts,
    })

    # Save to database
    try:
        await save_incident(
            db=db, room_id=room_id,
            threat_type=threat.get('type', 'UNKNOWN'),
            threat_level=threat.get('level', 'MEDIUM'),
            confidence=threat.get('confidence', 0.0),
            cctv_lat=cctv_coords[0], cctv_lon=cctv_coords[1],
            distance_m=dist_m, report_text=report,
        )
    except Exception as e:
        print(f'[DB] Save failed: {e}')
    print(f'[DISPATCH] {threat.get("type")} | {format_distance(dist_m)} | room {room_id}')


@app.websocket('/ws/{room_id}/{role}')
async def websocket_endpoint(ws: WebSocket, room_id: str, role: str,
                              db: AsyncSession = Depends(get_db)):
    if role not in ('cctv', 'police'):
        await ws.close(code=1008)
        return

    await manager.connect(room_id, role, ws)

    try:
        while True:
            raw   = await ws.receive_text()
            msg   = json.loads(raw)
            mtype = msg.get('type')

            if mtype == 'gps':
                lat = msg.get('lat', 0.0)
                lon = msg.get('lon', 0.0)
                gps_store.setdefault(room_id, {})[role] = (lat, lon)
                if role == 'cctv':
                    await manager.alert_police(room_id, {
                        'type': 'location_update',
                        'cctv_lat': lat, 'cctv_lon': lon,
                    })
                username = msg.get('username')
                if username:
                    try:
                        await update_user_gps(db, username, lat, lon)
                    except Exception: pass

            elif mtype == 'frame' and role == 'cctv':
                detection_ts = int(time.time() * 1000)
                loop   = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None, analyse_frame, msg.get('data', ''))

                # Always stream tracked objects to CCTV so bounding boxes appear instantly
                if result.get('tracked_persons') or result.get('threat_scores'):
                    await manager.send_to_cctv(room_id, {
                        'type':    'threat_scores',
                        'scores':   result.get('threat_scores', {}),
                        'tracked':  result.get('tracked_persons', []),
                    })

                reactive = classify_reactive_threat(
                    result.get('detections', []))
                if reactive and not is_on_cooldown(room_id, reactive['type']):
                    set_cooldown(room_id, reactive['type'])
                    await dispatch_alert(room_id, reactive, db, detection_ts, msg.get('data'))

                for pred in result.get('predictive_threats', []):
                    if not is_on_cooldown(room_id, pred['type']):
                        set_cooldown(room_id, pred['type'])
                        await dispatch_alert(room_id, pred, db, detection_ts, msg.get('data'))

                # ── Chain snatching alert (pose-based model) ───────────────
                chain_alert = result.get('chain_alert')
                if chain_alert and not is_on_cooldown(room_id, 'CHAIN_SNATCHING'):
                    set_cooldown(room_id, 'CHAIN_SNATCHING')
                    await dispatch_alert(room_id, chain_alert, db, detection_ts, msg.get('data'))

            elif mtype == 'audio' and role == 'cctv':
                detection_ts = int(time.time() * 1000)
                try:
                    audio_bytes = base64.b64decode(msg.get('data',''))
                    loop  = asyncio.get_event_loop()
                    threat = await loop.run_in_executor(
                        None, analyse_audio, audio_bytes)
                    if threat and not is_on_cooldown(room_id, threat['type']):
                        set_cooldown(room_id, threat['type'])
                        await dispatch_alert(room_id, threat, db, detection_ts)
                except Exception as e:
                    print(f'[AUDIO] Error: {e}')

            elif mtype == 'acknowledge' and role == 'police':
                print(f'[ACK] Officer in room {room_id} acknowledged')

    except WebSocketDisconnect:
        manager.disconnect(room_id, role)
    except Exception as e:
        print(f'[WS] Error in room {room_id}: {e}')
        manager.disconnect(room_id, role)


@app.get('/')
async def root():
    return {'status':'running','service':'Sentinel AI','version':'1.0.0'}

@app.get('/health')
async def health():
    return {'status':'healthy',
            'active_rooms': len(manager.rooms),
            'timestamp': datetime.datetime.utcnow().isoformat()}

@app.get('/rooms')
async def active_rooms():
    return {'rooms': manager.get_active_rooms()}

@app.get('/incidents')
async def list_incidents(room_id: str = None,
                          db: AsyncSession = Depends(get_db)):
    incidents = await get_incidents(db, room_id)
    return {'incidents': [
        {'id': i.id, 'room_id': i.room_id,
         'threat_type': i.threat_type, 'threat_level': i.threat_level,
         'confidence': i.confidence, 'distance_m': i.distance_m,
         'timestamp': i.timestamp.isoformat() if i.timestamp else None,
         'status': i.status}
        for i in incidents
    ]}

# Serve frontend static files at /app/
import os
from fastapi.staticfiles import StaticFiles

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, 'frontend')

if os.path.exists(FRONTEND_DIR):
    app.mount('/app', StaticFiles(directory=FRONTEND_DIR, html=True), name='frontend')
    print(f'[STATIC] SUCCESS: Frontend mounted at /app/ from {FRONTEND_DIR}')
else:
    print(f'[STATIC] ERROR: Frontend directory NOT FOUND at {FRONTEND_DIR}')
