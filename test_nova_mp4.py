import cv2
import boto3
from dotenv import load_dotenv

load_dotenv()

def convert_webm_to_mp4(in_file, out_file):
    cap = cv2.VideoCapture(in_file)
    if not cap.isOpened():
        print("Cannot open video")
        return False
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps: # NaN check
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

print("Re-encoding...")
if convert_webm_to_mp4("test.webm", "test.mp4"):
    print("Uploading to S3...")
    s3 = boto3.client("s3")
    s3.upload_file("test.mp4", "video-analyzer-nova", "nova_surveillance/test.mp4")
    
    print("Testing converse API with standard MP4...")
    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    s3_uri = "s3://video-analyzer-nova/nova_surveillance/test.mp4"
    try:
        response = client.converse(
            modelId="us.amazon.nova-lite-v1:0",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "video": {
                                "format": "mp4",
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
        print("Success:", response["output"]["message"]["content"][0]["text"])
    except Exception as e:
        print("Error:", e)
