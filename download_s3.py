import boto3
from dotenv import load_dotenv
load_dotenv()

client = boto3.client("s3", region_name="us-east-1")
client.download_file("video-analyzer-nova", "nova_surveillance/chunk_20260309_144153.webm", "test.webm")
print("Downloaded test.webm")
