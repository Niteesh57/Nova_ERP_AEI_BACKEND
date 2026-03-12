import boto3
from dotenv import load_dotenv
load_dotenv()

client = boto3.client("bedrock-runtime", region_name="us-east-1")
s3_uri = "s3://video-analyzer-nova/nova_surveillance/chunk_20260309_144153.webm"
try:
    response = client.converse(
        modelId="us.amazon.nova-2-lite-v1:0",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "video": {
                            "format": "webm",
                            "source": {
                                "s3Location": {"uri": s3_uri}
                            },
                        }
                    },
                    {"text": "Analyze this video."},
                ],
            }
        ],
    )
    print("Success:", response)
except Exception as e:
    print("Error:", e)
