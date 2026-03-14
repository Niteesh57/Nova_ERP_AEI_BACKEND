import asyncio
import cv2
import json
import os
import logging
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import WebSocket, UploadFile
from app.core.config import CAPTURE_INTERVAL_SECONDS
from app.core.models import Event
from app.services.s3_service import upload_video
from app.services.bedrock_service import analyze_video
from app.services.face_service import extract_faces_from_video
from app.services.chromadb_service import identify_face

logger = logging.getLogger(__name__)

EVENTS_LOG_FILE = "events_log.json"

def _convert_webm_to_mp4(in_file: str, out_file: str) -> bool:
    try:
        cap = cv2.VideoCapture(in_file)
        if not cap.isOpened():
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps != fps: # NaN check just in case
            fps = 15.0
            
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_file, fourcc, fps, (w, h))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
        cap.release()
        out.release()
        return True
    except Exception as e:
        logger.error(f"Failed to transcode {in_file} to mp4: {e}")
        return False


def _save_detection_to_db(payload: dict):
    """Synchronously insert a DetectionResult row into SQLite."""
    try:
        from app.db.database import SessionLocal
        from app.models.detection import DetectionResult
        db = SessionLocal()
        persons = payload.get("identified_persons") or []
        first_person = persons[0] if persons else {}
        row = DetectionResult(
            timestamp=payload.get("timestamp", ""),
            results_json=json.dumps(payload.get("results", {})),
            summary=payload.get("summary", ""),
            s3_uri=payload.get("s3_uri", ""),
            identified_name=first_person.get("name"),
            identified_email=first_person.get("email"),
            identified_persons_json=json.dumps(persons) if persons else None,
        )
        db.add(row)
        db.commit()
        db.refresh(row)
        row_id = row.id
        db.close()
        print(f"[DB] ✅ DetectionResult saved (id={row_id})")
    except Exception as e:
        print(f"[DB] ❌ Failed to save DetectionResult: {e}")
        logger.error(f"[DB] Failed to save DetectionResult: {e}", exc_info=True)


class SurveillanceManager:
    """Manages S3 uploads, Nova analysis, DB + JSON logging, and WebSocket broadcasting."""

    def __init__(self):
        self.running: bool = False
        self.events: List[Event] = []
        self.connected_clients: List[WebSocket] = []
        self.last_capture: Optional[str] = None

    # ── Event Management ──────────────────────────────────────────────

    def add_event(self, event: Event) -> bool:
        event.name = event.name.strip()
        if any(e.name == event.name for e in self.events):
            return False
        self.events.append(event)
        return True

    def remove_event(self, name: str) -> bool:
        name = name.strip()
        before = len(self.events)
        self.events = [e for e in self.events if e.name.strip() != name]
        return len(self.events) < before

    def list_events(self) -> List[Event]:
        return self.events

    def load_events_from_db(self):
        """Load persisted event triggers from SQLite into memory on startup."""
        try:
            from app.db.database import SessionLocal
            from app.models.event import EventTrigger
            db = SessionLocal()
            triggers = db.query(EventTrigger).all()
            db.close()
            for t in triggers:
                auth_emps = json.loads(t.authorized_employees) if t.authorized_employees else None
                evt = Event(name=t.name, description=t.description, authorized_employees=auth_emps)
                if not any(e.name == evt.name for e in self.events):
                    self.events.append(evt)
            print(f"[DB] Loaded {len(triggers)} event trigger(s) from DB")
        except Exception as e:
            print(f"[DB] Could not load event triggers: {e}")

    # ── WebSocket Client Management ──────────────────────────────────

    async def register(self, ws: WebSocket):
        await ws.accept()
        self.connected_clients.append(ws)

    def unregister(self, ws: WebSocket):
        if ws in self.connected_clients:
            self.connected_clients.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.connected_clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.unregister(ws)

    # ── Surveillance Status ──────────────────────────────────────────

    def set_running(self, running: bool):
        self.running = running
        print(f"[Manager] Surveillance running = {running}")

    def stop(self):
        self.running = False
        print("[Manager] Surveillance stopped.")

    # ── Video Processing Pipeline ────────────────────────────────────

    async def process_uploaded_video(self, file: UploadFile):
        if not self.running:
            print("[Manager] Received upload but surveillance was not started — auto-starting.")
            self.running = True

        self.last_capture = datetime.now(timezone.utc).isoformat()

        os.makedirs("tmp", exist_ok=True)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        extension = file.filename.split('.')[-1] if file.filename and '.' in file.filename else 'webm'
        local_video_path = f"tmp/chunk_{timestamp_str}.{extension}"
        local_upload_path = local_video_path
        s3_object_name = f"nova_surveillance/chunk_{timestamp_str}.{extension}"

        print(f"[Manager] Received upload: {file.filename}, saving to {local_video_path}")

        try:
            content = await file.read()
            print(f"[Manager] Read {len(content)} bytes.")
            with open(local_video_path, "wb") as f:
                f.write(content)

            # Transcode fragmented webm to compliant mp4 so Bedrock doesn't throw ValidationException
            local_mp4_path = local_video_path.replace(f'.{extension}', '.mp4')
            if extension.lower() in ['webm', 'mkv', 'ogg']:
                print(f"[Manager] Transcoding fragmented {extension} to compliant mp4...")
                loop = asyncio.get_event_loop()
                success = await loop.run_in_executor(None, _convert_webm_to_mp4, local_video_path, local_mp4_path)
                if success:
                    local_upload_path = local_mp4_path
                    s3_object_name = s3_object_name.replace(f'.{extension}', '.mp4')
                    print(f"[Manager] Transcode successful: {local_mp4_path}")
                else:
                    print(f"[Manager] Transcode failed, falling back to {local_video_path}")

            active_events = list(self.events)
            print(f"[Manager] Active events: {[e.name for e in active_events]}")

            if active_events:
                loop = asyncio.get_event_loop()
                s3_uri = await loop.run_in_executor(None, upload_video, local_upload_path, s3_object_name)

                if not s3_uri:
                    error_msg = {"error": "Failed to upload video to S3."}
                    await self.broadcast(error_msg)
                    return error_msg

                print(f"[Manager] S3 URI: {s3_uri}. Calling Amazon Nova...")

                result = await loop.run_in_executor(None, analyze_video, s3_uri, active_events)
                print(f"[Manager] Nova result: {result}")

                # ── Face Identification ──────────────────────────────────
                identified_persons = []
                event_results = result.get("results", {})
                
                # Check if ANY event returned > 0 rather than just boolean
                any_event_triggered = any(v for v in event_results.values() if v)

                if any_event_triggered:
                    print("[Manager] Event(s) triggered — attempting face identification...")
                    faces_bytes = await loop.run_in_executor(
                        None, extract_faces_from_video, local_video_path
                    )
                    
                    if faces_bytes:
                        for face in faces_bytes:
                            match = await loop.run_in_executor(None, identify_face, face)
                            if match:
                                identified_persons.append(match)
                                print(f"[Manager] 🙋 Identified person: {match['name']} ({match.get('email')})")
                            else:
                                identified_persons.append({"name": "Unknown Person", "email": None})
                                print("[Manager] 👤 Unknown person — no ChromaDB match")
                    else:
                        print("[Manager] 👤 No faces extracted from video")

                # ── Intrusion Alert Logic ────────────────────────────────
                alerts = []
                for event_name, is_detected in event_results.items():
                    val = is_detected > 0 if isinstance(is_detected, int) else bool(is_detected)
                    if val:
                        evt = next((e for e in active_events if e.name == event_name), None)
                        if evt and evt.authorized_employees:
                            # If there are NO faces, it's an unknown intrusion by default
                            if not identified_persons:
                                alert_msg = f"Intrusion Alert: '{event_name}' triggered by unknown/unseen person."
                                print(f"[ALERT] {alert_msg} -> (Mock) Sending Email to Admin...")
                                alerts.append(alert_msg)
                            else:
                                for person in identified_persons:
                                    if person["name"] not in evt.authorized_employees:
                                        alert_msg = f"Intrusion Alert: '{event_name}' triggered by unauthorized person: {person['name']}"
                                        print(f"[ALERT] {alert_msg} -> (Mock) Sending Email to Admin...")
                                        alerts.append(alert_msg)

                payload = {
                    "type": "event_result",
                    "timestamp": self.last_capture,
                    "results": event_results,
                    "summary": result.get("summary", ""),
                    "s3_uri": s3_uri,
                    "identified_persons": identified_persons,
                    "alerts": alerts,
                }

                # Save to JSON log
                self._append_to_log(payload)

                # Save to SQLite DB
                loop.run_in_executor(None, _save_detection_to_db, payload)

                # Broadcast to dashboard
                await self.broadcast(payload)
                return payload

            else:
                payload = {
                    "type": "heartbeat",
                    "timestamp": self.last_capture,
                    "message": "No events configured — video uploaded but not processed.",
                }
                await self.broadcast(payload)
                return payload

        except Exception as e:
            print(f"[Manager] ❌ Unexpected error: {e}")
            logger.error(f"Error processing video: {e}", exc_info=True)
            error_payload = {"error": str(e)}
            await self.broadcast(error_payload)
            return error_payload
        finally:
            for path in [local_video_path, local_video_path.replace(f'.{extension}', '.mp4')]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except Exception as e:
                        logger.error(f"Failed to cleanup {path}: {e}")

    def _append_to_log(self, data: dict):
        try:
            with open(EVENTS_LOG_FILE, "a") as f:
                f.write(json.dumps(data) + "\n")
            print(f"[Manager] ✅ Result appended to {EVENTS_LOG_FILE}")
        except Exception as e:
            print(f"[Manager] ❌ Failed to write log: {e}")


# Singleton instance
manager = SurveillanceManager()
