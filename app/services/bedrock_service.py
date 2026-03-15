import json
import re
import boto3
import logging
from typing import Dict, List
from app.core.config import AWS_REGION, BEDROCK_MODEL_ID
from app.core.models import Event

logger = logging.getLogger(__name__)


def _get_client():
    """Lazily creates Bedrock client so credentials are always fresh."""
    return boto3.client("bedrock-runtime", region_name=AWS_REGION)


def _extract_json(text: str) -> dict:
    """
    Robustly extract a JSON object from the model response.
    Handles markdown code fences and extra whitespace.
    """
    # Strip markdown code fences: ```json ... ``` or ``` ... ```
    stripped = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # Try direct parse first
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Try to find a JSON object anywhere in the response
    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError("No valid JSON found", text, 0)


def analyze_video(s3_uri: str, events: List[Event]) -> Dict:
    """
    Send an S3 video URI to Bedrock Amazon Nova and evaluate each event.
    Returns {"results": {event_name: bool}, "summary": str}.
    """
    if not events:
        return {}

    if not s3_uri:
        return {"results": {e.name: 0 for e in events}, "summary": "No S3 URI provided."}

    event_descriptions = "\n".join(
        f'- "{evt.name}": {evt.description}' for evt in events
    )

    prompt = (
        "You are a surveillance analysis system. Analyze the provided video "
        "and determine HOW MANY people are performing each of the following events. "
        "If the event did not occur, or 0 people are performing it, return 0.\n\n"
        f"Events to evaluate:\n{event_descriptions}\n\n"
        "Your ENTIRE response must be a single raw JSON object with NO markdown, NO code fences, "
        "NO extra text before or after. Use this exact structure where the value is an integer count:\n"
        '{"results": {"event_name": 1}, "summary": "One sentence summary."}'
    )

    video_format = s3_uri.split('.')[-1].lower() if '.' in s3_uri else 'mp4'
    if video_format not in ['mp4', 'mkv', 'webm', 'mov']:
        video_format = 'mp4'

    try:
        client = _get_client()
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "video": {
                                "format": video_format,
                                "source": {
                                    "s3Location": {"uri": s3_uri}
                                },
                            }
                        },
                        {"text": prompt},
                    ],
                }
            ],
        )

        raw_text = response["output"]["message"]["content"][0]["text"]
        print(f"[Bedrock] Raw response: {raw_text}")
        logger.info(f"[Bedrock] Raw response: {raw_text}")

        parsed = _extract_json(raw_text)
        results = parsed.get("results", {})
        summary = parsed.get("summary", "")

        # Ensure all events are represented
        for evt in events:
            if evt.name not in results:
                results[evt.name] = 0

        print(f"[Bedrock] ✅ Parsed results: {results}")
        return {"results": results, "summary": summary}

    except json.JSONDecodeError as e:
        print(f"[Bedrock] ❌ Could not parse JSON: {e}")
        logger.error(f"[Bedrock] JSON parse error: {e}")
        return {
            "results": {evt.name: 0 for evt in events},
            "summary": f"Could not parse model response as JSON: {str(e)}",
        }
    except Exception as e:
        print(f"[Bedrock] ❌ API error: {e}")
        logger.error(f"[Bedrock] API error: {e}", exc_info=True)
        return {
            "summary": f"API Error: {str(e)}",
        }

def chat_with_agent(user_query: str, session_history: List[Dict[str, str]] = None) -> str:
    """
    Conversational endpoint using Amazon Nova via Bedrock.
    """
    try:
        client = _get_client()
        
        system_prompt = (
            "You are Nova Sonnet, a highly capable AI assistant integrated into a video surveillance and ERP application. "
            "I can assist you with the knowledge graph and the tickets and tickets rising. Please feel free to reach out. "
            "Keep your responses relatively concise but helpful, as this may be spoken via Text-to-Speech."
        )

        # Basic integration of past messages into prompt
        messages = []
        if session_history:
            for msg in session_history:
                # Add past conversational blocks if full integration is needed later. Currently sticking to simple chat.
                pass
                
        messages.append({
            "role": "user",
            "content": [{"text": user_query}]
        })
        
        response = client.converse(
            modelId=BEDROCK_MODEL_ID,
            system=[{"text": system_prompt}],
            messages=messages,
        )
        
        reply_text = response["output"]["message"]["content"][0]["text"]
        return reply_text
    except Exception as e:
        logger.error(f"[Bedrock Agent] Failed to query LLM: {e}")
        raise e

