from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy.orm import Session
from pydantic import BaseModel

from app.db.database import get_db
from app.models.conversation import ConversationSession, ConversationMessage
from app.services.bedrock_service import chat_with_agent
from app.services.bedrock_streaming import BedrockWebsocketManager
import asyncio

router = APIRouter(prefix="/agent", tags=["Call Agent"])

class AgentChatRequest(BaseModel):
    session_id: str
    user_query: str

class AgentChatResponse(BaseModel):
    session_id: str
    user_query: str
    agent_response: str

@router.post("/chat", response_model=AgentChatResponse)
async def chat_endpoint(request: AgentChatRequest, db: Session = Depends(get_db)):
    """
    1. Sends the user_query to Bedrock (Nova Sonnet, etc).
    2. Logs the user_query and agent_response to the DB under session_id.
    3. Returns the agent_response and the session_id to the frontend.
    """
    
    # 1. Talk to Bedrock (or any LLM service)
    try:
        agent_response = chat_with_agent(request.user_query, session_history=[])
    except Exception as e:
        print(f"[Agent] Failed to chat with agent: {e}")
        raise HTTPException(status_code=500, detail=f"LLM Error: {str(e)}")

    # 2. Log to Database
    session = db.query(ConversationSession).filter(ConversationSession.session_id == request.session_id).first()
    if not session:
        session = ConversationSession(session_id=request.session_id)
        db.add(session)
        db.commit()
        db.refresh(session)
        
    new_message = ConversationMessage(
        session_id=session.session_id,
        user_query=request.user_query,
        agent_response=agent_response
    )
    
    db.add(new_message)
    db.commit()
    
    # 3. Return the payload
    return AgentChatResponse(
        session_id=session.session_id,
        user_query=request.user_query,
        agent_response=agent_response
    )

@router.websocket("/ws/stream")
async def websocket_stream_endpoint(websocket: WebSocket, session_id: str = ""):
    """
    Bidirectional audio streaming WebSocket.
    Browser sends 16kHz PCM audio bytes here -> We send to Bedrock
    Bedrock sends us 24kHz PCM audio -> We send back to Browser via WebSockets
    All conversation turns are automatically logged to the SQL DB under session_id.
    """
    await websocket.accept()

    # Use a provided session_id, or generate a new one for this call session
    if not session_id:
        session_id = str(__import__('uuid').uuid4())
    print(f"[AgentWS] Starting session: {session_id}")

    manager = BedrockWebsocketManager(websocket, session_id=session_id)
    
    # 1. Initialize AWS Bedrock Stream
    success = await manager.initialize_stream()
    if not success:
        await websocket.close(code=1011)
        return

    # 2. Receive frames from Browser
    try:
        while manager.is_active:
            # We expect raw binary audio chunks (or JSON for control messages)
            message = await websocket.receive()
            
            if message["type"] == "websocket.disconnect":
                print("[AgentWS] Received websocket.disconnect from client")
                break
                
            if "bytes" in message and message["bytes"]:
                await manager.process_incoming_audio(message["bytes"])
            elif "text" in message:
                print(f"[AgentWS] Received text: {message['text']}")

    except WebSocketDisconnect:
        print("[AgentWS] Client disconnected via WebSocketDisconnect exception")
    except Exception as e:
        print(f"[AgentWS] Error in receive loop: {type(e).__name__}: {e}")
    finally:
        print("[AgentWS] Closing manager and cleaning up stream...")
        await manager.close()
        print("[AgentWS] Cleanup complete.")
