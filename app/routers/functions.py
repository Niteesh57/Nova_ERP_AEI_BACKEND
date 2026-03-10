"""
functions.py — Three AI Code Agents (Code Generation, Code Scanner, Cost Optimizer)

Endpoints:
  GET  /functions/user-stories       → all user stories (for the Attach picker)
  POST /functions/invoke              → invoke selected Bedrock agent (SSE streaming)
"""
import json
import logging
import uuid
import boto3
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.models.user_story import UserStory
from app.models.product import Product

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/functions", tags=["Functions"])

# ── AWS Bedrock Agent Config ─────────────────────────────────────────────────

AGENT_CONFIG = {
    "code_generation": {
        "agentId": "FPCWMWCCLS",
        "agentAliasId": "TSTALIASID",
        "label": "Code Generation",
    },
    "code_scanner": {
        "agentId": "FU0GLWNVOX",
        "agentAliasId": "TSTALIASID",
        "label": "Code Scanner",
    },
    "cost_optimizer": {
        "agentId": "77Y0L0HTU6",
        "agentAliasId": "TSTALIASID",
        "label": "Cost Optimizer",
    },
}


import botocore.config

def get_bedrock_agent_client():
    return boto3.client(
        "bedrock-agent-runtime",
        region_name="us-east-1",
        config=botocore.config.Config(
            read_timeout=600,        # 10 minutes
            connect_timeout=10,
            retries={"max_attempts": 0},  # don't retry on timeout
        ),
    )


# ── Schemas ──────────────────────────────────────────────────────────────────

class InvokeRequest(BaseModel):
    agent_type: str          # "code_generation" | "code_scanner" | "cost_optimizer"
    prompt: str              # full user message (may include attached story context)
    session_id: Optional[str] = None  # pass same value for multi-turn conversations


class StoryBrief(BaseModel):
    id: int
    product_id: int
    product_name: str
    title: str
    description: Optional[str] = None
    tag: Optional[str] = None

    class Config:
        from_attributes = True


# ── Endpoints ────────────────────────────────────────────────────────────────

@router.get("/user-stories", response_model=List[StoryBrief])
def get_all_user_stories(db: Session = Depends(get_db)):
    """Return all user stories with their parent product name, for the Attach picker."""
    stories = (
        db.query(UserStory, Product.idea_name.label("product_name"))
        .join(Product, UserStory.product_id == Product.id)
        .order_by(Product.idea_name, UserStory.id)
        .all()
    )

    result = []
    for story, product_name in stories:
        result.append(StoryBrief(
            id=story.id,
            product_id=story.product_id,
            product_name=product_name,
            title=story.title,
            description=story.description,
            tag=story.tag,
        ))
    return result


@router.post("/invoke")
def invoke_agent(payload: InvokeRequest):
    """
    Invoke the selected Bedrock Agent and return the response as Server-Sent Events.
    Pass the same session_id across turns to maintain conversation context.

    If the agent requires confirmation for an action group function, it will emit a
    `returnControl` event which we forward as {"type": "confirm_required", ...}.
    The frontend must then call /functions/confirm with the user's decision.
    """
    config = AGENT_CONFIG.get(payload.agent_type)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown agent_type: {payload.agent_type}")

    session_id = payload.session_id or str(uuid.uuid4())

    def stream_response():
        # First SSE: send the session_id so the frontend can persist it
        yield f"data: {json.dumps({'type': 'session_id', 'session_id': session_id})}\n\n"

        try:
            client = get_bedrock_agent_client()
            response = client.invoke_agent(
                agentId=config["agentId"],
                agentAliasId=config["agentAliasId"],
                sessionId=session_id,
                inputText=payload.prompt,
                enableTrace=False,
            )

            event_stream = response.get("completion", [])
            for event in event_stream:
                logger.debug(f"[Functions] Agent event keys: {list(event.keys())}")

                # ── Normal text chunk ──────────────────────────────────────
                chunk = event.get("chunk", {})
                if "bytes" in chunk:
                    text = chunk["bytes"].decode("utf-8")
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"

                # ── Action group confirmation required ─────────────────────
                elif "returnControl" in event:
                    rc = event["returnControl"]
                    invocation_id = rc.get("invocationId", "")
                    invocation_inputs = rc.get("invocationInputs", [])

                    # Parse inputs for display
                    functions_info = []
                    for inp in invocation_inputs:
                        fn_inp = inp.get("functionInvocationInput", {})
                        functions_info.append({
                            "actionGroup": fn_inp.get("actionGroup", ""),
                            "function": fn_inp.get("function", ""),
                            "parameters": fn_inp.get("parameters", []),
                        })

                    logger.info(f"[Functions] returnControl invocationId={invocation_id}, functions={functions_info}")
                    yield f"data: {json.dumps({'type': 'confirm_required', 'invocation_id': invocation_id, 'invocation_inputs': invocation_inputs, 'functions': functions_info})}\n\n"

                # ── Trace or other events (skip silently) ──────────────────
                else:
                    pass

        except Exception as e:
            logger.error(f"[Functions] Bedrock Agent invoke error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        # Signal the end of the stream
        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


class ConfirmRequest(BaseModel):
    agent_type: str
    session_id: str
    invocation_id: str
    invocation_inputs: list            # the raw invocationInputs from the returnControl event
    confirmed: bool                    # True = user clicked Confirm, False = Deny


@router.post("/confirm")
def confirm_agent_action(payload: ConfirmRequest):
    """
    Resume a Bedrock Agent session after a returnControl event.
    Sends the user's Confirm/Deny answer back to the agent and streams the response.
    """
    config = AGENT_CONFIG.get(payload.agent_type)
    if not config:
        raise HTTPException(status_code=400, detail=f"Unknown agent_type: {payload.agent_type}")

    def stream_response():
        try:
            client = get_bedrock_agent_client()

            # Build the returnControlInvocationResults based on the user's choice
            return_results = []
            for inp in payload.invocation_inputs:
                fn_inp = inp.get("functionInvocationInput", {})
                action_group = fn_inp.get("actionGroup", "")
                function_name = fn_inp.get("function", "")
                status_msg = "CONFIRMED" if payload.confirmed else "DENIED"

                return_results.append({
                    "functionResult": {
                        "actionGroup": action_group,
                        "function": function_name,
                        "responseBody": {
                            "TEXT": {
                                "body": f"User {status_msg.lower()} this action."
                            }
                        },
                        "confirmationState": "CONFIRM" if payload.confirmed else "DENY",
                    }
                })

            session_state = {
                "invocationId": payload.invocation_id,
                "returnControlInvocationResults": return_results,
            }

            logger.info(f"[Functions] Resuming session={payload.session_id} confirmed={payload.confirmed} sessionState={session_state}")

            response = client.invoke_agent(
                agentId=config["agentId"],
                agentAliasId=config["agentAliasId"],
                sessionId=payload.session_id,
                inputText="",
                sessionState=session_state,
                enableTrace=False,
            )

            event_stream = response.get("completion", [])
            for event in event_stream:
                chunk = event.get("chunk", {})
                if "bytes" in chunk:
                    text = chunk["bytes"].decode("utf-8")
                    yield f"data: {json.dumps({'type': 'text', 'content': text})}\n\n"
                elif "returnControl" in event:
                    # Nested confirm (multi-step tools) - forward again
                    rc = event["returnControl"]
                    invocation_id = rc.get("invocationId", "")
                    invocation_inputs = rc.get("invocationInputs", [])
                    functions_info = []
                    for inp in invocation_inputs:
                        fn_inp = inp.get("functionInvocationInput", {})
                        functions_info.append({
                            "actionGroup": fn_inp.get("actionGroup", ""),
                            "function": fn_inp.get("function", ""),
                            "parameters": fn_inp.get("parameters", []),
                        })
                    yield f"data: {json.dumps({'type': 'confirm_required', 'invocation_id': invocation_id, 'invocation_inputs': invocation_inputs, 'functions': functions_info})}\n\n"

        except Exception as e:
            logger.error(f"[Functions] Confirm error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        stream_response(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
