import os
from dotenv import load_dotenv

load_dotenv()

# AWS Configuration
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
BEDROCK_MODEL_ID = os.getenv(
    "BEDROCK_MODEL_ID",
    "us.amazon.nova-lite-v1:0"
)
AWS_S3_BUCKET = os.getenv("AWS_S3_BUCKET", "")

if not AWS_S3_BUCKET:
    print("WARNING: AWS_S3_BUCKET is not set in environment or .env file.")

# Surveillance Configuration
CAPTURE_INTERVAL_SECONDS = int(os.getenv("CAPTURE_INTERVAL_SECONDS", "30"))
WEBCAM_INDEX = int(os.getenv("WEBCAM_INDEX", "0"))
