import numpy as np
import io
import wave

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    yamnet_model   = hub.load('https://tfhub.dev/google/yamnet/1')
    class_map_path = yamnet_model.class_map_path().numpy().decode()
    class_names    = [
        l.strip().split(',')[2].strip('"')
        for l in tf.io.gfile.GFile(class_map_path).readlines()[1:]
    ]
    YAMNET_AVAILABLE = True
    print('[YAMNET] Audio model loaded')
except Exception as e:
    YAMNET_AVAILABLE = False
    print(f'[YAMNET] Not available: {e}')

THREAT_SOUNDS = {
    'Screaming':       ('SCREAMING',    'HIGH',   0.40),
    'Shout':           ('SHOUTING',     'MEDIUM', 0.40),
    'Crying, sobbing': ('CRYING',       'MEDIUM', 0.40),
    'Glass':           ('GLASS_BREAK',  'HIGH',   0.50),
    'Gunshot':         ('GUNSHOT',      'HIGH',   0.60),
    'Explosion':       ('EXPLOSION',    'HIGH',   0.60),
    'Fighting':        ('FIGHT_AUDIO',  'HIGH',   0.45),
}


def analyse_audio(audio_blob: bytes) -> dict | None:
    if not YAMNET_AVAILABLE:
        return None
    try:
        wf       = wave.open(io.BytesIO(audio_blob))
        frames   = wf.readframes(wf.getnframes())
        waveform = (np.frombuffer(frames, dtype=np.int16)
                    .astype(np.float32) / 32768.0)
        if len(waveform) < 1000:
            return None
        scores, _, _ = yamnet_model(waveform)
        mean_scores  = np.mean(scores, axis=0)
        for keyword, (threat_type, level, threshold) in THREAT_SOUNDS.items():
            matching = [i for i, name in enumerate(class_names)
                        if keyword.lower() in name.lower()]
            if not matching: continue
            best = float(max(mean_scores[i] for i in matching))
            if best >= threshold:
                return {'type': threat_type, 'level': level,
                        'confidence': round(best, 3),
                        'detail': keyword, 'source': 'audio'}
    except Exception as e:
        print(f'[YAMNET] Error: {e}')
    return None
