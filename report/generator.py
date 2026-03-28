import httpx
import datetime

OLLAMA_URL   = 'http://localhost:11434/api/generate'
OLLAMA_MODEL = 'mistral'


def _build_prompt(event: dict, distance_m: int) -> str:
    t = datetime.datetime.utcnow().strftime('%H:%M:%S UTC')
    predictive = event.get('type') in \
                 ('PREDICTED_THREAT', 'BEHAVIOUR_WARNING')
    if predictive:
        return f'''You are an AI police intelligence assistant.
Suspicious behaviour flagged BEFORE any crime occurred.
Type: {event.get('type')}  Score: {event.get('score','N/A')}/100
Distance: {distance_m}m  Time: {t}
Write: 1) Pattern description 2) Risk level 3) Three pre-emptive actions.
Advisory tone. Under 150 words.'''
    return f'''You are an AI police dispatch assistant.
Generate a formal emergency incident report.
Threat: {event.get('type')}  Level: {event.get('level')}
Detail: {event.get('detail','N/A')}
Confidence: {round(event.get('confidence',0)*100,1)}%
Distance: {distance_m}m  Time: {t}
Write: 1) Situation summary 2) Risk level 3) Four immediate actions.
Professional tone. Under 180 words.'''


async def generate_report(event: dict, distance_m: int) -> str:
    prompt = _build_prompt(event, distance_m)
    try:
        async with httpx.AsyncClient(timeout=45.0) as client:
            resp = await client.post(OLLAMA_URL, json={
                'model':  OLLAMA_MODEL,
                'prompt': prompt,
                'stream': False,
                'options': {'temperature': 0.3, 'num_predict': 250}
            })
            resp.raise_for_status()
            return resp.json().get('response', 'Report generation failed.')
    except httpx.ConnectError:
        return (f'[OFFLINE] Threat: {event.get("type")} | '
                f'Level: {event.get("level")} | '
                f'Distance: {distance_m}m | Start ollama serve')
    except Exception as e:
        return f'[ERROR] {event.get("type")} detected. {distance_m}m away. {e}'
