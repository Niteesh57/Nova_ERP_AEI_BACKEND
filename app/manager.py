import asyncio
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

logger = logging.getLogger(__name__)

EVENTS_LOG_FILE = "events_log.json"


def _save_detection_to_db(payload: dict):
    """Synchronously insert a DetectionResult row into SQLite."""
    try:
        from app.db.database import SessionLocal
        from app.models.detection import DetectionResult
        db = SessionLocal()
        row = DetectionResult(
            timestamp=payload.get("timestamp", ""),
            results_json=json.dumps(payload.get("results", {})),
            summary=payload.get("summary", ""),
            s3_uri=payload.get("s3_uri", ""),
        )
        db.add(row)
        db.commit()
        db.close()
        print(f"[DB] ✅ DetectionResult saved (id={row.id})")
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
        if any(e.name == event.name for e in self.events):
            return False
        self.events.append(event)
        return True

    def remove_event(self, name: str) -> bool:
        before = len(self.events)
        self.events = [e for e in self.events if e.name != name]
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
                evt = Event(name=t.name, description=t.description)
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
        s3_object_name = f"nova_surveillance/chunk_{timestamp_str}.{extension}"

        print(f"[Manager] Received upload: {file.filename}, saving to {local_video_path}")

        try:
            content = await file.read()
            print(f"[Manager] Read {len(content)} bytes.")
            with open(local_video_path, "wb") as f:
                f.write(content)

            active_events = list(self.events)
            print(f"[Manager] Active events: {[e.name for e in active_events]}")

            if active_events:
                loop = asyncio.get_event_loop()
                s3_uri = await loop.run_in_executor(None, upload_video, local_video_path, s3_object_name)

                if not s3_uri:
                    error_msg = {"error": "Failed to upload video to S3."}
                    await self.broadcast(error_msg)
                    return error_msg

                print(f"[Manager] S3 URI: {s3_uri}. Calling Amazon Nova...")

                result = await loop.run_in_executor(None, analyze_video, s3_uri, active_events)
                print(f"[Manager] Nova result: {result}")

                payload = {
                    "type": "event_result",
                    "timestamp": self.last_capture,
                    "results": result.get("results", {}),
                    "summary": result.get("summary", ""),
                    "s3_uri": s3_uri,
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
            if os.path.exists(local_video_path):
                try:
                    os.remove(local_video_path)
                except Exception as e:
                    logger.error(f"Failed to cleanup {local_video_path}: {e}")

    def _append_to_log(self, data: dict):
        try:
            with open(EVENTS_LOG_FILE, "a") as f:
                f.write(json.dumps(data) + "\n")
            print(f"[Manager] ✅ Result appended to {EVENTS_LOG_FILE}")
        except Exception as e:
            print(f"[Manager] ❌ Failed to write log: {e}")


# Singleton instance
manager = SurveillanceManager()
