import asyncio
import base64
import json
import uuid
import pytz
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

# ── Helpers ───────────────────────────────────────────────────────────────────

def query_knowledge_base(query_text: str) -> str:
    """Query the AWS Knowledge Base using retrieve() (Hybrid Approach 3)."""
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
            return ""
        passages = []
        for idx, r in enumerate(results):
            content = r.get("content", {}).get("text", "")
            if content:
                passages.append(f"[{idx+1}] {content}")
        return "\n".join(passages)
    except Exception as e:
        debug_print(f"KB Search Error: {e}")
        return ""


def raise_ticket_in_db(product_name: str, issue_description: str, username: str) -> str:
    """Create a support ticket in the database."""
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
        return f"Ticket #{ticket.id}"
    except Exception as e:
        db.rollback()
        debug_print(f"Raise Ticket DB Error: {e}")
        return f"Error: {e}"
    finally:
        db.close()


# ── Bedrock Manager ───────────────────────────────────────────────────────────

class BedrockWebsocketManager:
    """
    Manages bidirectional streaming with AWS Bedrock Nova 2 Sonic.
    KB and Ticket integration uses the Hybrid Approach (no tool-use in promptStart).
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

        self.response_task   = None
        self.stream_response = None
        self.is_active       = False
        self.bedrock_client  = None

        self.prompt_name        = str(uuid.uuid4())
        self.sys_content_name   = str(uuid.uuid4())
        self.audio_content_name = str(uuid.uuid4())

        self._user_text_buf:  list = []
        self._agent_text_buf: list = []
        self._current_role:   str | None = None

        self._kb_injected  = False
        self._audio_paused = False  # True during KB injection to block stale audio chunks
        self._ticket_raised = False

        self._full_user_transcript = ""  # Persistent across the whole session

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
                        "voiceId": "matthew",
                        "encoding": "base64",
                        "audioType": "SPEECH"
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
                f"You are Nova Sonnet, a professional AI customer support agent for the following products: {product_list_str}. "
                "Keep your responses extremely short and concise (1-2 sentences maximum). "
                "Start by greeting the user and asking 'How can I help you today?'. "
                "Listen carefully for the product name and problem they describe. "
                "CRITICAL: If the product is not in the list of products above, firmly state that we only support those products and do not assist further. "
                "When knowledge base information is injected into this conversation, use it to help the user. "
                "If the issue remains unresolved for a supported product, offer to raise a support ticket. "
                "Ask if they want to provide their name and email (tell them this is optional). "
                "Once they answer, confirm the ticket has been created and close the conversation warmly."
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
        if not self.is_active or self._audio_paused:
            return  # drop audio during KB injection (avoids stale content name)
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

    # ── Hybrid KB & Ticket injection ──────────────────────────────────────────

    def _is_issue_query(self, text: str) -> bool:
        keywords = [
            "issue", "problem", "error", "bug", "fix", "debug", "not working",
            "doesn't work", "crash", "fail", "broken", "help", "troubleshoot",
            "ticket", "raise", "can't", "cannot", "unable"
        ]
        lower = text.lower()
        return any(kw in lower for kw in keywords)

    async def _inject_kb_context(self, query: str):
        """
        Hybrid Approach 3: 
        1. Close the open audio content block (required by Nova 2 Sonic).
        2. Inject KB context as a new SYSTEM text turn.
        3. Reopen a fresh audio content block so the user can continue speaking.

        Nova 2 Sonic raises InvalidEventBytes if you open a new content block
        while the audio block is still active.
        """
        debug_print(f"KB inject start: {query[:80]}")

        kb_text = await asyncio.get_event_loop().run_in_executor(
            None, query_knowledge_base, query
        )
        if not kb_text:
            debug_print("KB returned nothing useful")
            return

        if not self.is_active:
            return

        self._audio_paused = True  # pause audio pump to avoid stale content name
        inject_cn = str(uuid.uuid4())
        context_msg = (
            f"[Knowledge Base context for this conversation]\n{kb_text}\n"
            f"[Use the above to answer the user. If this doesn't resolve their issue, "
            f"ask for their name/email and tell them a support ticket will be raised.]"
        )

        # 1. Close the currently-open audio content block
        await self.send_raw_event(self._build_content_end(self.audio_content_name))
        await asyncio.sleep(0.1)

        # 2. Inject KB context as an ASSISTANT turn.
        # Nova 2 Sonic only allows ONE SYSTEM block per prompt — the system prompt is
        # already sent at init. We inject KB findings as an ASSISTANT "internal note".
        context_msg = (
            f"I found the following relevant information in the knowledge base:\n\n"
            f"{kb_text}\n\n"
            f"I will use this to help the user. If it does not resolve their issue, "
            f"I will ask for their name and email and offer to raise a support ticket."
        )
        await self.send_raw_event(self._build_text_content_start(inject_cn, "ASSISTANT"))
        await self.send_raw_event(self._build_text_input(inject_cn, context_msg))
        await self.send_raw_event(self._build_content_end(inject_cn))
        await asyncio.sleep(0.1)

        # 3. Reopen a fresh audio content block
        self.audio_content_name = str(uuid.uuid4())
        await self.send_raw_event(self._build_audio_content_start())
        self._audio_paused = False  # resume audio pump with new content name

        debug_print("KB context injected and audio stream reopened")

    async def _check_and_raise_ticket(self):
        """
        Heuristic approach to raise a ticket without requiring Bedrock tool-use.
        Parses the user buffer for an email address and context.
        """
        if self._ticket_raised:
            return

        user_history = self._full_user_transcript.lower()
        
        # Look for an email address
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', user_history)
        if not email_match:
            # Sometimes users say "at gmail dot com"
            email_match = re.search(r'[\w\.-]+\s+(?:at|@)\s+[\w\.-]+\s+(?:dot|\.)\s+\w+', user_history)
            
        username = email_match.group(0) if email_match else "Unknown User"
        
        username = email_match.group(0) if email_match else "Unknown User"
        
        # Try to guess product dynamically based on available_products
        product = None
        if self.available_products:
            # 1. Exact or partial substring match of the whole product name (ignoring case)
            for p in self.available_products:
                if p.lower() in user_history:
                    product = p
                    break
            
            # 2. Heuristic word matching if strict substring doesn't work
            if not product:
                for p in self.available_products:
                    # e.g., 'Nova HealthLens (AI Health Monitoring System)' -> words longer than 3 chars
                    words = [w for w in re.findall(r'\w+', p.lower()) if len(w) > 3 
                             and w not in ['system', 'platform', 'app', 'application', 'software', 'management', 'monitoring']]
                    for w in words:
                        if w in user_history:
                            product = p
                            break
                    if product:
                        break

        if not product:
            debug_print("No supported product detected. Skipping DB ticket creation.")
            self._ticket_raised = True
            return
        # The issue is ONLY what the user said
        user_only_history = ' '.join(self._user_text_buf).lower()
        if len(user_only_history) == 0:
            user_only_history = user_history # fallback
            
        issue = user_only_history[-500:] if len(user_only_history) > 500 else user_only_history
        
        self._ticket_raised = True
        debug_print(f"Triggering background ticket creation for {username}")
        
        loop = asyncio.get_running_loop()
        try:
            # Shield the ticket creation task so it finishes even if the stream closes
            await asyncio.shield(loop.run_in_executor(
                None, raise_ticket_in_db, product, issue, username
            ))
        except Exception as e:
            debug_print(f"Executor failed to run ticket DB insert: {e}")

    # ── Response processor ────────────────────────────────────────────────────

    async def _process_responses(self):
        try:
            while self.is_active:
                output = await self.stream_response.await_output()
                result = await output[1].receive()
                if not (result.value and result.value.bytes_):
                    continue
                data = result.value.bytes_.decode('utf-8')
                try:
                    j = json.loads(data)
                    if 'event' not in j:
                        continue
                    event = j['event']

                    if 'audioOutput' in event:
                        audio_bytes = base64.b64decode(event['audioOutput']['content'])
                        await self.websocket.send_bytes(audio_bytes)

                    elif 'textOutput' in event:
                        txt  = event['textOutput']['content']
                        role = event['textOutput']['role']
                        debug_print(f"Text [{role}]: {txt[:80]}")

                        if '{ "interrupted" : true }' in txt:
                            await self._flush_turn_to_db()
                            await self.websocket.send_json({"type": "interrupt"})
                        else:
                            if self._current_role and self._current_role != role:
                                if self._current_role == 'ASSISTANT':
                                    await self._flush_turn_to_db()
                            self._current_role = role
                            if role == 'USER':
                                self._user_text_buf.append(txt)
                                self._full_user_transcript += " " + txt
                                combined = self._full_user_transcript.strip().lower()

                                # Trigger KB injection once when issue is detected
                                if not self._kb_injected and self._is_issue_query(combined) and len(combined) > 40:
                                    self._kb_injected = True  # set sync to prevent double-inject race
                                    asyncio.create_task(self._inject_kb_context(combined))
                            else:
                                self._agent_text_buf.append(txt)
                                combined_agent = ' '.join(self._agent_text_buf).lower()
                                combined_user = self._full_user_transcript.strip().lower()
                                
                                # Heuristically detect ticket creation: 
                                # 1. Agent talks about resolving/ticket/reaching out.
                                # 2. User has already provided an email address.
                                if not self._ticket_raised:
                                    # Email is optional now, focus heavily on the agent confirming creation
                                    # We just check if the agent mentions "ticket" and a creation verb anywhere in its response.
                                    has_ticket_word = "ticket" in combined_agent
                                    has_creation_verb = any(word in combined_agent for word in ["created", "raised", "logged", "opened"])
                                    agent_confirming = has_ticket_word and has_creation_verb
                                    
                                    # Debug log every time the agent speaks to see what we're evaluating
                                    if len(combined_agent) > 10:
                                        debug_print(f"[Heuristic Check] agent_confirming: {agent_confirming}, user_len: {len(combined_user)}")
                                    
                                    if agent_confirming and len(combined_user) > 30:
                                        asyncio.create_task(self._check_and_raise_ticket())
                                        
                            await self.websocket.send_json({"type": "text", "role": role, "content": txt})

                except json.JSONDecodeError:
                    debug_print(f"JSON decode error: {data[:80]}")

        except Exception as e:
            err = str(e)
            ignored = ("ValidationException", "Stream closed", "CANCELLED", "InvalidStateError")
            if not any(x in err for x in ignored):
                debug_print(f"AWS Read Error: {type(e).__name__}: {err}")
        finally:
            self.is_active = False
            try:
                await self.websocket.close(code=1000)
            except Exception:
                pass

    # ── DB logging ────────────────────────────────────────────────────────────

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

        await self._flush_turn_to_db()

        # Wait for the reader task to fully stop before closing the stream
        if self.response_task and not self.response_task.done():
            self.response_task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(self.response_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                pass

        # Send graceful termination events directly (bypassing send_raw_event is_active guard)
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
                except Exception:
                    pass
            try:
                await self.stream_response.input_stream.close()
            except Exception as e:
                debug_print(f"Stream close: {type(e).__name__}")
