import asyncio
import base64
import json
import uuid
import pytz
import datetime
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
from fastapi import WebSocket
from sqlalchemy.orm import Session as DBSession
from app.db.database import SessionLocal
from app.models.conversation import ConversationSession, ConversationMessage

DEBUG = True

def debug_print(message):
    if DEBUG:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] BEDROCK: {message}")

class ToolProcessor:
    def __init__(self):
        self.tasks = {}
    
    async def process_tool_async(self, tool_name, tool_content):
        task_id = str(uuid.uuid4())
        task = asyncio.create_task(self._run_tool(tool_name, tool_content))
        self.tasks[task_id] = task
        try:
            return await task
        finally:
            if task_id in self.tasks:
                del self.tasks[task_id]
    
    async def _run_tool(self, tool_name, tool_content):
        tool = tool_name.lower()
        if tool == "getdateandtimetool":
            pst_timezone = pytz.timezone("America/Los_Angeles")
            pst_date = datetime.datetime.now(pst_timezone)
            return {
                "formattedTime": pst_date.strftime("%I:%M %p"),
                "date": pst_date.strftime("%Y-%m-%d"),
                "timezone": "PST"
            }
        else:
            return {"error": f"Unsupported tool: {tool_name}"}

class BedrockWebsocketManager:
    """Manages bidirectional streaming with AWS Bedrock bridged to a FastAPI WebSocket"""
    
    START_SESSION_EVENT = '{"event": {"sessionStart": {"inferenceConfiguration": {"maxTokens": 1024, "topP": 0.9, "temperature": 0.7}}}}'
    CONTENT_START_EVENT = '{"event": {"contentStart": {"promptName": "%s", "contentName": "%s", "type": "AUDIO", "interactive": true, "role": "USER", "audioInputConfiguration": {"mediaType": "audio/lpcm", "sampleRateHertz": 16000, "sampleSizeBits": 16, "channelCount": 1, "audioType": "SPEECH", "encoding": "base64"}}}}'
    AUDIO_EVENT_TEMPLATE = '{"event": {"audioInput": {"promptName": "%s", "contentName": "%s", "content": "%s"}}}'
    TEXT_CONTENT_START_EVENT = '{"event": {"contentStart": {"promptName": "%s", "contentName": "%s", "type": "TEXT", "role": "%s", "interactive": false, "textInputConfiguration": {"mediaType": "text/plain"}}}}'
    TEXT_INPUT_EVENT = '{"event": {"textInput": {"promptName": "%s", "contentName": "%s", "content": "%s"}}}'
    CONTENT_END_EVENT = '{"event": {"contentEnd": {"promptName": "%s", "contentName": "%s"}}}'
    PROMPT_END_EVENT = '{"event": {"promptEnd": {"promptName": "%s"}}}'
    SESSION_END_EVENT = '{"event": {"sessionEnd": {}}}'
    
    def __init__(self, websocket: WebSocket, session_id: str, model_id='amazon.nova-2-sonic-v1:0', region='us-east-1'):
        self.websocket = websocket
        self.session_id = session_id
        self.model_id = model_id
        self.region = region
        
        self.response_task = None
        self.stream_response = None
        self.is_active = False
        self.bedrock_client = None
        
        self.prompt_name = str(uuid.uuid4())
        self.content_name = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self.tool_processor = ToolProcessor()
        self.pending_tool_tasks = {}

        # Text accumulation buffers for DB logging
        self._user_text_buf: list[str] = []
        self._agent_text_buf: list[str] = []
        self._current_role: str | None = None

    def _initialize_client(self):
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)
        
    def _get_prompt_start(self):
        get_default_tool_schema = json.dumps({"type": "object", "properties": {}, "required": []})
        event = {
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {"mediaType": "application/json"},
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "getDateAndTimeTool",
                                    "description": "get information about the current date and time",
                                    "inputSchema": {"json": get_default_tool_schema}
                                }
                            }
                        ]
                    }
                }
            }
        }
        return json.dumps(event)

    async def initialize_stream(self):
        if not self.bedrock_client:
            self._initialize_client()
        
        try:
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )
            self.is_active = True
            
            # Send initialization payload
            default_system_prompt = "You are Nova Sonnet, a professional, human-sounding AI assistant."
            
            init_events = [
                self.START_SESSION_EVENT,
                self._get_prompt_start(),
                self.TEXT_CONTENT_START_EVENT % (self.prompt_name, self.content_name, "SYSTEM"),
                self.TEXT_INPUT_EVENT % (self.prompt_name, self.content_name, default_system_prompt),
                self.CONTENT_END_EVENT % (self.prompt_name, self.content_name),
                self.CONTENT_START_EVENT % (self.prompt_name, self.audio_content_name) # Open the mic
            ]
            
            for event in init_events:
                await self.send_raw_event(event)
                await asyncio.sleep(0.05)
                
            # Start background reader
            self.response_task = asyncio.create_task(self._process_responses())
            debug_print("Bedrock stream connected successfully")
            return True
            
        except Exception as e:
            self.is_active = False
            debug_print(f"Failed to init Bedrock: {str(e)}")
            return False

    async def send_raw_event(self, event_json: str):
        if not self.stream_response or not self.is_active:
            return
        
        event = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        try:
            await self.stream_response.input_stream.send(event)
        except Exception as e:
            debug_print(f"AWS Send Error: {e}")

    async def process_incoming_audio(self, audio_bytes: bytes):
        """Called externally by the websocket loop to pump mic data to AWS"""
        if not self.is_active: return
        try:
            b64_str = base64.b64encode(audio_bytes).decode('utf-8')
            evt = self.AUDIO_EVENT_TEMPLATE % (self.prompt_name, self.audio_content_name, b64_str)
            await self.send_raw_event(evt)
        except Exception as e:
            debug_print(f"Audio Pump Error: {e}")

    async def _process_responses(self):
        """Reads AWS Bedrock outputs and sends them to the browser via WebSocket"""
        try:
            while self.is_active:
                output = await self.stream_response.await_output()
                result = await output[1].receive()
                if result.value and result.value.bytes_:
                    data = result.value.bytes_.decode('utf-8')
                    try:
                        j = json.loads(data)
                        if 'event' in j:
                            event = j['event']
                            
                            if 'contentStart' in event:
                                debug_print("Bedrock triggered contentStart")

                            # Forward Audio Down
                            if 'audioOutput' in event:
                                audio_payload = event['audioOutput']['content']
                                audio_bytes = base64.b64decode(audio_payload)
                                await self.websocket.send_bytes(audio_bytes)
                                
                            # Forward Text Signals Down
                            elif 'textOutput' in event:
                                txt = event['textOutput']['content']
                                role = event['textOutput']['role']
                                debug_print(f"Bedrock sent text [{role}]: {txt}")

                                # ─ Accumulate text for DB logging ─
                                if '{ "interrupted" : true }' in txt:
                                    # Treat as turn boundary – flush what we have
                                    await self._flush_turn_to_db()
                                    await self.websocket.send_json({"type": "interrupt"})
                                else:
                                    # Detect role switch – that means previous turn is complete
                                    if self._current_role and self._current_role != role:
                                        if self._current_role == 'ASSISTANT':
                                            # We finished an assistant turn, flush user+agent pair
                                            await self._flush_turn_to_db()
                                    self._current_role = role
                                    if role == 'USER':
                                        self._user_text_buf.append(txt)
                                    else:
                                        self._agent_text_buf.append(txt)
                                    await self.websocket.send_json({"type": "text", "role": role, "content": txt})

                            elif 'toolUse' in event:
                                debug_print(f"Tool use requested: {event['toolUse']}")
                                
                    except json.JSONDecodeError:
                        debug_print(f"Failed to decode AWS json: {data}")
        except Exception as e:
            if "ValidationException" not in str(e) and "Stream closed" not in str(e):
                debug_print(f"AWS Read Error: {e}")
        finally:
            self.is_active = False
            # Ensure WS is notified
            try:
                await self.websocket.close(code=1000)
            except: pass

    async def _flush_turn_to_db(self):
        """Write the accumulated USER+ASSISTANT turn to the SQL database."""
        user_text = ' '.join(self._user_text_buf).strip()
        agent_text = ' '.join(self._agent_text_buf).strip()

        # Only log if we have both sides of the conversation
        if user_text and agent_text:
            debug_print(f"Logging turn to DB | User: '{user_text[:60]}...' | Agent: '{agent_text[:60]}...'")
            try:
                db: DBSession = SessionLocal()
                try:
                    # Upsert Session
                    session = db.query(ConversationSession).filter_by(session_id=self.session_id).first()
                    if not session:
                        # Calculate next NOVA ERP (N) number
                        count = db.query(ConversationSession).count()
                        session_name = f"NOVA ERP ({count + 1})"
                        session = ConversationSession(session_id=self.session_id, session_name=session_name)
                        db.add(session)
                        db.commit()
                        db.refresh(session)
                        debug_print(f"Created session '{session_name}'")
                    # Insert Message
                    msg = ConversationMessage(
                        session_id=self.session_id,
                        user_query=user_text,
                        agent_response=agent_text
                    )
                    db.add(msg)
                    db.commit()
                    debug_print("DB write successful")
                finally:
                    db.close()
            except Exception as e:
                debug_print(f"DB write error: {e}")
        else:
            debug_print("Skipping DB flush, both sides not populated yet")

        # Clear buffers for next turn
        self._user_text_buf = []
        self._agent_text_buf = []
        self._current_role = None

    async def close(self):
        if not self.is_active: return
        self.is_active = False
        
        # Flush any pending turn before closing
        await self._flush_turn_to_db()

        for task in self.pending_tool_tasks.values(): task.cancel()
        if self.response_task and not self.response_task.done(): self.response_task.cancel()
        
        await self.send_raw_event(self.CONTENT_END_EVENT % (self.prompt_name, self.audio_content_name))
        await self.send_raw_event(self.PROMPT_END_EVENT % self.prompt_name)
        await self.send_raw_event(self.SESSION_END_EVENT)
        
        if self.stream_response:
            try: await self.stream_response.input_stream.close()
            except: pass
