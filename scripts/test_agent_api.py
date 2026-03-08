import httpx
import asyncio

async def main():
    print("Testing Lead Agent API...")
    async with httpx.AsyncClient(timeout=300) as client:
        response = await client.post(
            "http://127.0.0.1:8000/leads/start",
            json={"goal": "Best SEO agencies in London"}
        )
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

if __name__ == "__main__":
    asyncio.run(main())
