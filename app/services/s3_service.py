import boto3
import logging
from app.core.config import AWS_REGION, AWS_S3_BUCKET

logger = logging.getLogger(__name__)


def _get_client():
    """Lazily creates the S3 client so that credentials are always read freshly."""
    return boto3.client("s3", region_name=AWS_REGION)


def upload_video(file_path: str, object_name: str) -> str:
    """
    Uploads a video file to S3 and returns the s3:// URI on success, or '' on failure.
    """
    if not AWS_S3_BUCKET:
        logger.error("[S3] AWS_S3_BUCKET is not set — cannot upload.")
        print("[S3] ERROR: AWS_S3_BUCKET is not set in .env")
        return ""

    print(f"[S3] Uploading '{file_path}' → s3://{AWS_S3_BUCKET}/{object_name} (region={AWS_REGION})")
    logger.info(f"[S3] Uploading '{file_path}' → s3://{AWS_S3_BUCKET}/{object_name}")

    try:
        client = _get_client()
        client.upload_file(file_path, AWS_S3_BUCKET, object_name)
        uri = f"s3://{AWS_S3_BUCKET}/{object_name}"
        print(f"[S3] ✅ Upload successful: {uri}")
        logger.info(f"[S3] Upload successful: {uri}")
        return uri
    except Exception as e:
        print(f"[S3] ❌ Upload failed: {e}")
        logger.error(f"[S3] Upload failed: {e}", exc_info=True)
        return ""
