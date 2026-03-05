from fastapi import APIRouter, UploadFile, File
from app.manager import manager
from app.core.models import SurveillanceStatus
from app.core.config import CAPTURE_INTERVAL_SECONDS

router = APIRouter(prefix="/surveillance", tags=["Surveillance Control"])

@router.post("/start")
async def start_surveillance():
    """Tells backend the frontend started recording."""
    if manager.running:
        return {"detail": "Surveillance is already running"}
    manager.set_running(True)
    return {"detail": "Surveillance started"}

@router.post("/stop")
async def stop_surveillance():
    """Tells backend the frontend stopped recording."""
    if not manager.running:
        return {"detail": "Surveillance is not running"}
    manager.set_running(False)
    return {"detail": "Surveillance stopped"}

@router.get("/status", response_model=SurveillanceStatus)
async def surveillance_status():
    """Get current surveillance status."""
    return SurveillanceStatus(
        running=manager.running,
        active_events=len(manager.events),
        capture_interval=CAPTURE_INTERVAL_SECONDS,
        last_capture=manager.last_capture,
    )

@router.post("/upload")
async def upload_video_endpoint(file: UploadFile = File(...)):
    """Frontend uploads the 30s chunk here."""
    print(f"[Route] /surveillance/upload called — filename={file.filename}, content_type={file.content_type}")
    return await manager.process_uploaded_video(file)

@router.get("/debug")
async def debug_info():
    """Returns current pipeline state for debugging."""
    from app.core.config import AWS_REGION, AWS_S3_BUCKET, BEDROCK_MODEL_ID
    import os
    return {
        "running": manager.running,
        "active_events": [e.name for e in manager.events],
        "last_capture": manager.last_capture,
        "aws_region": AWS_REGION,
        "s3_bucket": AWS_S3_BUCKET,
        "bedrock_model": BEDROCK_MODEL_ID,
        "log_file_exists": os.path.exists("events_log.json"),
        "tmp_files": os.listdir("tmp") if os.path.exists("tmp") else [],
    }
