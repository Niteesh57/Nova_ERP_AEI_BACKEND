import asyncio
import base64
import json
import uuid
import datetime
import re
import boto3
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.models import InvokeModelWithBidirectionalStreamInputChunk, BidirectionalInputPayloadPart
from aws_sdk_bedrock_runtime.config import Config
from smithy_aws_core.identity.environment import EnvironmentCredentialsResolver
from fastapi import WebSocket
from sqlalchemy.orm import Session as DBSession
from app.db.database import SessionLocal
from app.models.conversation import ConversationSession, ConversationMessage
from app.models.ticket import Ticket
from app.models.product import Product
from app.core.config import AWS_REGION

DEBUG = True

def debug_print(message):
    if DEBUG:
        print(f"[{datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] BEDROCK: {message}")


# ── Tool Implementations ───────────────────────────────────────────────────────

def _query_knowledge_base(query_text: str) -> str:
    """Query the AWS Knowledge Base using retrieve()."""
    kb_id = "FB5BW6TSAA"
    try:
        client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
        response = client.retrieve(
            knowledgeBaseId=kb_id,
            retrievalQuery={"text": query_text},
            retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": 3}}
        )
        results = response.get("retrievalResults", [])
        if not results:
            return "No relevant information found in the knowledge base."
        passages = []
        for idx, r in enumerate(results):
            content = r.get("content", {}).get("text", "")
            if content:
                passages.append(f"[{idx+1}] {content}")
        return "\n".join(passages)
    except Exception as e:
        debug_print(f"KB Search Error: {e}")
        return f"Knowledge base search failed: {str(e)}"


def _raise_ticket_in_db(product_name: str, issue_description: str, username: str) -> str:
    """Create a support ticket in the database and return the ticket ID."""
    db: DBSession = SessionLocal()
    try:
        ticket = Ticket(
            product_name=product_name,
            user_facing_issue=issue_description,
            username=username
        )
        db.add(ticket)
        db.commit()
        db.refresh(ticket)
        debug_print(f"Ticket successfully created: ID {ticket.id}")
        return f"Ticket #{ticket.id} successfully created."
    except Exception as e:
        db.rollback()
        debug_print(f"Raise Ticket DB Error: {e}")
        return f"Failed to create ticket: {str(e)}"
    finally:
        db.close()


# ── Tool Processor ─────────────────────────────────────────────────────────────

class ToolProcessor:
    """Handles async execution of tool calls without blocking the audio stream."""

    async def process_tool_async(self, tool_name: str, tool_input: dict) -> dict:
        tool = tool_name.lower()
        debug_print(f"Processing tool: {tool_name} with input: {tool_input}")

        if tool == "searchknowledgebasetool":
            query = tool_input.get("query", "")
            loop = asyncio.get_event_loop()
            result_text = await loop.run_in_executor(None, _query_knowledge_base, query)
            return {"result": result_text}

        elif tool == "raisetickettool":
            product  = tool_input.get("product_name", "Unknown Product")
            issue    = tool_input.get("issue_description", "No description provided")
            username = tool_input.get("username", "Anonymous")
            loop = asyncio.get_event_loop()
            result_text = await loop.run_in_executor(None, _raise_ticket_in_db, product, issue, username)
            return {"result": result_text}

        else:
            return {"error": f"Unknown tool: {tool_name}"}


# ── Bedrock Manager ───────────────────────────────────────────────────────────

class BedrockWebsocketManager:
    """
    Manages bidirectional streaming with AWS Bedrock Nova 2 Sonic.
    Uses native tool-use for KB search and ticket creation.
    Includes a decoupled audio sender loop for stutter-free playback.
    """

    _START_SESSION = json.dumps({
        "event": {
            "sessionStart": {
                "inferenceConfiguration": {
                    "maxTokens": 1024,
                    "topP": 0.9,
                    "temperature": 0.7
                }
            }
        }
    })
    _SESSION_END = '{"event": {"sessionEnd": {}}}'

    def __init__(self, websocket: WebSocket, session_id: str,
                 model_id='amazon.nova-2-sonic-v1:0', region='us-east-1'):
        self.websocket  = websocket
        self.session_id = session_id
        self.model_id   = model_id
        self.region     = region
        self.available_products = []

        self.response_task: asyncio.Task | None = None
        self.stream_response = None
        self.is_active       = False
        self.bedrock_client  = None

        self.prompt_name        = str(uuid.uuid4())
        self.sys_content_name   = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        # Transcript buffering for DB logging
        self._user_text_buf:  list[str] = []
        self._agent_text_buf: list[str] = []
        self._current_role:   str | None = None

        # Tool-use state (populated from incoming events)
        self._tool_name:    str = ""
        self._tool_use_id:  str = ""
        self._tool_input:   str = ""  # accumulated JSON string

        # Pending async tool tasks
        self._pending_tool_tasks: dict[str, asyncio.Task] = {}
        self._tool_processor = ToolProcessor()
        self._tool_active = False

        # Audio chunk queue — decouples receive loop from WebSocket send
        self._audio_queue: asyncio.Queue = asyncio.Queue(maxsize=300)
        self._audio_sender_task: asyncio.Task | None = None

    # ── Client ───────────────────────────────────────────────────────────────

    def _initialize_client(self):
        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=EnvironmentCredentialsResolver(),
        )
        self.bedrock_client = BedrockRuntimeClient(config=config)

    # ── Event builders ────────────────────────────────────────────────────────

    def _build_prompt_start(self):
        """Build promptStart with toolConfiguration for KB search and ticketing."""
        kb_search_schema = json.dumps({
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up in the knowledge base."
                }
            },
            "required": ["query"]
        })

        raise_ticket_schema = json.dumps({
            "type": "object",
            "properties": {
                "product_name": {
                    "type": "string",
                    "description": "The name of the product the user is having issues with."
                },
                "issue_description": {
                    "type": "string",
                    "description": "A detailed description of the user's issue."
                },
                "username": {
                    "type": "string",
                    "description": "The user's name or email address. Use 'Anonymous' if not provided."
                }
            },
            "required": ["product_name", "issue_description", "username"]
        })

        return json.dumps({
            "event": {
                "promptStart": {
                    "promptName": self.prompt_name,
                    "textOutputConfiguration": {"mediaType": "text/plain"},
                    "audioOutputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 24000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "voiceId": "tiffany",
                        "encoding": "base64",
                        "audioType": "SPEECH"
                    },
                    "toolUseOutputConfiguration": {
                        "mediaType": "application/json"
                    },
                    "toolConfiguration": {
                        "tools": [
                            {
                                "toolSpec": {
                                    "name": "searchKnowledgeBaseTool",
                                    "description": (
                                        "Retrieve official information or troubleshooting steps for our software products. "
                                        "Call this tool whenever a user asks a general question about a product "
                                        "or describes a specific technical problem."
                                    ),
                                    "inputSchema": {"json": kb_search_schema}
                                }
                            },
                            {
                                "toolSpec": {
                                    "name": "raiseTicketTool",
                                    "description": (
                                        "Create a support ticket in the database for unresolved customer issues. "
                                        "Call this tool only when the customer explicitly agrees to raise a ticket. "
                                        "Collect relevant details (product, issue, username) first if not already known."
                                    ),
                                    "inputSchema": {"json": raise_ticket_schema}
                                }
                            }
                        ]
                    }
                }
            }
        })

    def _build_text_content_start(self, content_name: str, role: str):
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "type": "TEXT",
                    "role": role,
                    "interactive": False,
                    "textInputConfiguration": {"mediaType": "text/plain"}
                }
            }
        })

    def _build_text_input(self, content_name: str, text: str):
        return json.dumps({
            "event": {
                "textInput": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": text
                }
            }
        })

    def _build_content_end(self, content_name: str):
        return json.dumps({
            "event": {
                "contentEnd": {
                    "promptName": self.prompt_name,
                    "contentName": content_name
                }
            }
        })

    def _build_audio_content_start(self):
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": self.audio_content_name,
                    "type": "AUDIO",
                    "interactive": True,
                    "role": "USER",
                    "audioInputConfiguration": {
                        "mediaType": "audio/lpcm",
                        "sampleRateHertz": 16000,
                        "sampleSizeBits": 16,
                        "channelCount": 1,
                        "audioType": "SPEECH",
                        "encoding": "base64"
                    }
                }
            }
        })

    def _build_tool_content_start(self, content_name: str, tool_use_id: str):
        return json.dumps({
            "event": {
                "contentStart": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "interactive": False,
                    "type": "TOOL",
                    "role": "USER",
                    "toolResultInputConfiguration": {
                        "toolUseId": tool_use_id,
                        "type": "TEXT",
                        "textInputConfiguration": {"mediaType": "text/plain"}
                    }
                }
            }
        })

    def _build_tool_result(self, content_name: str, result: dict):
        return json.dumps({
            "event": {
                "toolResult": {
                    "promptName": self.prompt_name,
                    "contentName": content_name,
                    "content": json.dumps(result)
                }
            }
        })

    # ── Stream lifecycle ──────────────────────────────────────────────────────

    async def initialize_stream(self):
        if not self.bedrock_client:
            self._initialize_client()
        try:
            self.stream_response = await self.bedrock_client.invoke_model_with_bidirectional_stream(
                InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
            )
            self.is_active = True

            # Fetch products dynamically from DB
            db = SessionLocal()
            try:
                products = db.query(Product.idea_name).all()
                self.available_products = [p[0] for p in products]
            except Exception as e:
                debug_print(f"Error fetching products: {e}")
                self.available_products = []
            finally:
                db.close()

            product_list_str = ", ".join(self.available_products) if self.available_products else "our software products"

            system_prompt = (
                f"You are Nova Sonnet, a professional AI customer support agent for: {product_list_str}. "
                "I can assist you with the knowledge graph and the tickets and tickets rising. Please feel free to reach out. "
                "Keep responses extremely short (2 sentences max). "
                "CRITICAL: If a product is mentioned that is NOT in the list above, inform the user we don't support it. "
                "Whenever a user asks about a supported product or describes a problem, you MUST call 'searchKnowledgeBaseTool'. "
                "Do not answer based on general knowledge for product-specific details. "
                "If searching KB doesn't resolve the issue, offer to raise a ticket via 'raiseTicketTool'. "
                "Confirm ticket creation verbally once done."
            )

            for evt in [
                self._START_SESSION,
                self._build_prompt_start(),
                self._build_text_content_start(self.sys_content_name, "SYSTEM"),
                self._build_text_input(self.sys_content_name, system_prompt),
                self._build_content_end(self.sys_content_name),
                self._build_audio_content_start(),
            ]:
                await self.send_raw_event(evt)
                await asyncio.sleep(0.05)

            # Start decoupled audio sender and response reader
            self._audio_sender_task = asyncio.create_task(self._audio_sender_loop())
            self.response_task = asyncio.create_task(self._process_responses())
            debug_print("Bedrock stream connected successfully")
            return True

        except Exception as e:
            self.is_active = False
            debug_print(f"Failed to init Bedrock: {e}")
            return False

    async def send_raw_event(self, event_json: str):
        if not self.stream_response or not self.is_active:
            return
        chunk = InvokeModelWithBidirectionalStreamInputChunk(
            value=BidirectionalInputPayloadPart(bytes_=event_json.encode('utf-8'))
        )
        try:
            await self.stream_response.input_stream.send(chunk)
        except Exception as e:
            debug_print(f"AWS Send Error: {e}")

    async def process_incoming_audio(self, audio_bytes: bytes):
        if not self.is_active or self._tool_active:
            return
        try:
            b64 = base64.b64encode(audio_bytes).decode('utf-8')
            await self.send_raw_event(json.dumps({
                "event": {
                    "audioInput": {
                        "promptName": self.prompt_name,
                        "contentName": self.audio_content_name,
                        "content": b64
                    }
                }
            }))
        except Exception as e:
            debug_print(f"Audio Pump Error: {e}")

    # ── Dedicated audio sender loop ───────────────────────────────────────────

    async def _audio_sender_loop(self):
        """Drains the audio queue and sends to WebSocket to prevent stalls."""
        while self.is_active or not self._audio_queue.empty():
            try:
                audio_bytes = await asyncio.wait_for(
                    self._audio_queue.get(), timeout=0.5
                )
                await self.websocket.send_bytes(audio_bytes)
                self._audio_queue.task_done()
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                debug_print(f"Audio sender error: {type(e).__name__}: {e}")
                break

    # ── Tool Handling ─────────────────────────────────────────────────────────

    def handle_tool_request(self, tool_name: str, tool_input: dict, tool_use_id: str):
        """Kick off async tool execution — non-blocking."""
        content_name = str(uuid.uuid4())
        task = asyncio.create_task(
            self._execute_tool_and_send_result(tool_name, tool_input, tool_use_id, content_name)
        )
        self._pending_tool_tasks[content_name] = task
        task.add_done_callback(lambda t: self._pending_tool_tasks.pop(content_name, None))

    async def _execute_tool_and_send_result(
        self, tool_name: str, tool_input: dict, tool_use_id: str, content_name: str
    ):
        """Execute tool, send result to Bedrock, and notify frontend."""
        try:
            debug_print(f"Executing tool: {tool_name}")
            self._tool_active = True

            # Notify frontend that tool is running
            await self._ws_send_json({"type": "tool_status", "tool": tool_name, "status": "running"})

            result = await self._tool_processor.process_tool_async(tool_name, tool_input)

            # Nova 2 Sonic requires closing the current block before starting a new one.
            # 1. Close the previous audio input block
            await self.send_raw_event(self._build_content_end(self.audio_content_name))
            await asyncio.sleep(0.1)

            # 2. Send tool result block
            await self.send_raw_event(self._build_tool_content_start(content_name, tool_use_id))
            await self.send_raw_event(self._build_tool_result(content_name, result))
            await self.send_raw_event(self._build_content_end(content_name))
            await asyncio.sleep(0.1)

            # 3. Re-open audio input block for continued interaction
            self.audio_content_name = str(uuid.uuid4())
            await self.send_raw_event(self._build_audio_content_start())

            debug_print(f"Tool complete: {tool_name}")
            # Notify frontend that tool is done
            await self._ws_send_json({"type": "tool_status", "tool": tool_name, "status": "done"})

        except Exception as e:
            debug_print(f"Tool error [{tool_name}]: {e}")
            try:
                # Attempt graceful error recovery
                err_res = {"error": f"Tool failed: {str(e)}"}
                await self.send_raw_event(self._build_content_end(self.audio_content_name))
                await asyncio.sleep(0.05)
                await self.send_raw_event(self._build_tool_content_start(content_name, tool_use_id))
                await self.send_raw_event(self._build_tool_result(content_name, err_res))
                await self.send_raw_event(self._build_content_end(content_name))
                await asyncio.sleep(0.05)
                self.audio_content_name = str(uuid.uuid4())
                await self.send_raw_event(self._build_audio_content_start())
                await self._ws_send_json({"type": "tool_status", "tool": tool_name, "status": "error"})
            except:
                pass
        finally:
            self._tool_active = False

    # ── Response Processor ────────────────────────────────────────────────────

    async def _process_responses(self):
        try:
            while self.is_active:
                try:
                    output = await self.stream_response.await_output()
                    result = await output[1].receive()
                except (asyncio.CancelledError, Exception) as e:
                    if self.is_active:
                        debug_print(f"Reader receive error: {e}")
                    break

                if not (result.value and result.value.bytes_):
                    continue

                data = result.value.bytes_.decode('utf-8')
                try:
                    j = json.loads(data)
                    if 'event' not in j: continue
                    event = j['event']

                    # Audio output -> enqueue for sender loop
                    if 'audioOutput' in event:
                        audio_bytes = base64.b64decode(event['audioOutput']['content'])
                        try:
                            self._audio_queue.put_nowait(audio_bytes)
                        except asyncio.QueueFull:
                            debug_print("Audio queue full - dropping chunk")

                    # Text output -> buffer + send to browser
                    elif 'textOutput' in event:
                        txt  = event['textOutput']['content']
                        role = event['textOutput']['role']
                        debug_print(f"Text [{role}]: {txt[:80]}")

                        if '{ "interrupted" : true }' in txt:
                            await self._flush_turn_to_db()
                            # Clear audio queue on interruption
                            while not self._audio_queue.empty():
                                try:
                                    self._audio_queue.get_nowait()
                                    self._audio_queue.task_done()
                                except asyncio.QueueEmpty: break
                            await self._ws_send_json({"type": "interrupt"})
                        else:
                            if self._current_role and self._current_role != role:
                                if self._current_role == 'ASSISTANT':
                                    await self._flush_turn_to_db()
                            self._current_role = role
                            
                            # Handle cumulative vs delta text
                            target_buf = self._user_text_buf if role == 'USER' else self._agent_text_buf
                            current_full = "".join(target_buf)
                            
                            if txt.startswith(current_full) and len(txt) > len(current_full):
                                # It's cumulative! Append only the new part.
                                delta = txt[len(current_full):]
                                target_buf.append(delta)
                            elif not current_full.endswith(txt):
                                # It's likely a delta or a fresh start.
                                target_buf.append(txt)

                            await self._ws_send_json({"type": "text", "role": role, "content": "".join(target_buf)})

                    # toolUse -> accumulate info
                    elif 'toolUse' in event:
                        tu = event['toolUse']
                        self._tool_name   = tu.get('toolName', '')
                        self._tool_use_id = tu.get('toolUseId', '')
                        self._tool_input  = tu.get('content', '{}')

                    # contentEnd -> check for TOOL
                    elif 'contentEnd' in event:
                        ce = event['contentEnd']
                        if ce.get('type') == 'TOOL':
                            try:
                                t_input = json.loads(self._tool_input) if self._tool_input else {}
                            except: t_input = {}
                            self.handle_tool_request(self._tool_name, t_input, self._tool_use_id)
                            # Reset tool state
                            self._tool_name = ""; self._tool_use_id = ""; self._tool_input = ""

                    elif 'completionEnd' in event:
                        await self._flush_turn_to_db()

                    elif 'contentStart' in event:
                        self._current_role = event['contentStart'].get('role')

                except json.JSONDecodeError:
                    pass

        except Exception as e:
            err = str(e)
            ignored = ("ValidationException", "Stream closed", "CANCELLED", "InvalidStateError", "StopAsyncIteration")
            if not any(x in err for x in ignored):
                debug_print(f"AWS Read Error: {type(e).__name__}: {err}")
        finally:
            self.is_active = False
            try:
                await self.websocket.close(code=1000)
            except: pass

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _ws_send_json(self, payload: dict):
        try:
            await self.websocket.send_json(payload)
        except: pass

    async def _flush_turn_to_db(self):
        user_text  = ' '.join(self._user_text_buf).strip()
        agent_text = ' '.join(self._agent_text_buf).strip()
        if user_text and agent_text:
            try:
                db: DBSession = SessionLocal()
                try:
                    session = db.query(ConversationSession).filter_by(session_id=self.session_id).first()
                    if not session:
                        count = db.query(ConversationSession).count()
                        session = ConversationSession(
                            session_id=self.session_id,
                            session_name=f"NOVA ERP ({count + 1})"
                        )
                        db.add(session)
                        db.commit()
                        db.refresh(session)
                    db.add(ConversationMessage(
                        session_id=self.session_id,
                        user_query=user_text,
                        agent_response=agent_text
                    ))
                    db.commit()
                    debug_print("DB write OK")
                finally:
                    db.close()
            except Exception as e:
                debug_print(f"DB write error: {e}")
        self._user_text_buf  = []
        self._agent_text_buf = []
        self._current_role   = None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    async def close(self):
        if not self.is_active:
            return
        self.is_active = False

        # Cancel pending tool tasks
        if hasattr(self, '_pending_tool_tasks'):
            for task in list(self._pending_tool_tasks.values()):
                task.cancel()

        await self._flush_turn_to_db()

        # Wait gracefully for reader task
        res_task = self.response_task
        if res_task and not res_task.done():
            try:
                await asyncio.wait_for(asyncio.shield(res_task), timeout=0.5)
            except:
                if not res_task.done():
                    res_task.cancel()

        # Wait for audio sender loop
        sender_task = self._audio_sender_task
        if sender_task and not sender_task.done():
            sender_task.cancel()

        # Send graceful termination events directly
        if self.stream_response:
            for evt in [
                self._build_content_end(self.audio_content_name),
                json.dumps({"event": {"promptEnd": {"promptName": self.prompt_name}}}),
                self._SESSION_END,
            ]:
                try:
                    chunk = InvokeModelWithBidirectionalStreamInputChunk(
                        value=BidirectionalInputPayloadPart(bytes_=evt.encode('utf-8'))
                    )
                    await self.stream_response.input_stream.send(chunk)
                except:
                    pass
            try:
                await self.stream_response.input_stream.close()
            except:
                pass
