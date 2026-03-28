from fastapi import WebSocket
from typing import Dict


class RoomManager:
    def __init__(self):
        # room_id -> { 'cctv': WebSocket, 'police': WebSocket }
        self.rooms: Dict[str, Dict[str, WebSocket]] = {}

    async def connect(self, room_id: str, role: str, ws: WebSocket):
        await ws.accept()
        if room_id not in self.rooms:
            self.rooms[room_id] = {}
        self.rooms[room_id][role] = ws
        print(f'[ROOM] {role.upper()} connected to room {room_id}')

    def disconnect(self, room_id: str, role: str):
        if room_id in self.rooms:
            self.rooms[room_id].pop(role, None)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
        print(f'[ROOM] {role.upper()} disconnected from room {room_id}')

    async def alert_police(self, room_id: str, payload: dict):
        room = self.rooms.get(room_id, {})
        police_ws = room.get('police')
        if police_ws:
            try:
                await police_ws.send_json(payload)
                print(f'[ALERT] Sent to police in room {room_id}')
            except Exception as e:
                print(f'[ALERT] Failed: {e}')
        else:
            print(f'[ALERT] No police in room {room_id}')

    async def send_to_cctv(self, room_id: str, payload: dict):
        room = self.rooms.get(room_id, {})
        cctv_ws = room.get('cctv')
        if cctv_ws:
            try:
                await cctv_ws.send_json(payload)
            except Exception as e:
                print(f'[CCTV] Failed: {e}')

    def get_active_rooms(self):
        return {
            room_id: list(roles.keys())
            for room_id, roles in self.rooms.items()
        }


# Single instance used across the entire app
manager = RoomManager()
