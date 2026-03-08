import asyncio
import json
import os
import sys
import traceback
from typing import Optional, TypedDict
from datetime import datetime

import boto3
from langgraph.graph import StateGraph, END
from nova_act import NovaAct, workflow

from app.db.database import SessionLocal
from app.models.leads import LeadSession, Lead, LeadLog

# ─── AWS Clients ─────────────────────────────────────────────────────────────

def _bedrock():
    return boto3.client("bedrock-runtime", region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))

def _nova_lite(prompt: str) -> str:
    """Call Bedrock Nova Lite."""
    client = _bedrock()
    body = json.dumps({
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": 1024, "temperature": 0.1}
    })
    resp = client.invoke_model(modelId="amazon.nova-lite-v1:0", body=body)
    result = json.loads(resp["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


# ─── DB & SSE Helpers ────────────────────────────────────────────────────────

def _log_to_db(db_session_id: int, step: str, message: str, queue=None):
    db = SessionLocal()
    try:
        log = LeadLog(session_id=db_session_id, step=step, message=message)
        db.add(log)
        db.commit()
    finally:
        db.close()
    if queue:
        try:
            queue.put_nowait({"step": step, "message": message})
        except asyncio.QueueFull:
            pass

def _save_lead(db_session_id: int, url: Optional[str], reasoning: Optional[str]):
    db = SessionLocal()
    try:
        lead = Lead(
            session_id=db_session_id,
            url=url,
            reasoning=reasoning
        )
        db.add(lead)
        db.commit()
        db.refresh(lead)
    finally:
        db.close()

def _mark_session(db_session_id: int, status: str):
    db = SessionLocal()
    try:
        s = db.query(LeadSession).get(db_session_id)
        if s:
            s.status = status
            s.updated_at = datetime.utcnow()
            db.commit()
    finally:
        db.close()

_SESSION_QUEUES: dict[int, asyncio.Queue] = {}

def get_or_create_queue(session_id: int) -> asyncio.Queue:
    if session_id not in _SESSION_QUEUES:
        _SESSION_QUEUES[session_id] = asyncio.Queue(maxsize=500)
    return _SESSION_QUEUES[session_id]

def cleanup_queue(session_id: int):
    _SESSION_QUEUES.pop(session_id, None)


# ─── State Graph ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    db_session_id: int
    goal: str
    refined_query: str
    raw_results: str
    log_queue: object

def refiner_node(state: AgentState) -> AgentState:
    """Refines the user's raw goal into an optimized Google Search Query."""
    goal = state["goal"]
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    _log_to_db(db_session_id, "planner", f"Refining user query: '{goal}'", queue)
    
    prompt = f"""
    Turn the following user request into a highly optimized Google search query to find the best relevant business websites.
    Do not add extra conversational text. Return ONLY the search string itself.
    User Request: {goal}
    """
    refined_query = _nova_lite(prompt)
    
    # Clean up quotes if LLM added them
    refined_query = refined_query.strip('"').strip("'")
    
    _log_to_db(db_session_id, "planner", f"Optimized query: '{refined_query}'", queue)
    state["refined_query"] = refined_query
    return state

def executor_node(state: AgentState) -> AgentState:
    """Uses Tavily API to search Google and reliably get URLs."""
    import httpx
    query = state["refined_query"]
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    _log_to_db(db_session_id, "executor", f"Calling Tavily Search API...", queue)
    
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        err = "TAVILY_API_KEY not found in environment."
        _log_to_db(db_session_id, "executor", err, queue)
        state["raw_results"] = f"Error: {err}"
        return state
        
    try:
        response = httpx.post(
            "https://api.tavily.com/search",
            json={
                "api_key": tavily_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 10
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()
        
        # Format the results into a string for the extractor
        results_text = []
        for r in data.get("results", []):
            results_text.append(f"URL: {r.get('url')}\nTitle: {r.get('title')}\nContent: {r.get('content')}\n---")
            
        observation = "\n".join(results_text)
        _log_to_db(db_session_id, "executor", f"Tavily returned {len(data.get('results', []))} results.", queue)
        state["raw_results"] = observation
    except Exception as e:
        _log_to_db(db_session_id, "executor", f"Tavily API Error: {str(e)}", queue)
        state["raw_results"] = f"Error: {e}"
        
    return state

def extractor_node(state: AgentState) -> AgentState:
    """Parses raw text into structured JSON URLs and Reasoning without hallucinating."""
    raw_results = state.get("raw_results", "")
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    _log_to_db(db_session_id, "critic", "Parsing raw results into structured JSON...", queue)
    
    prompt = f"""
    You are extracting real search results. Look at the text below. 
    Find the actual, absolute URLs (they must start with http or https and must NOT be example.com or yoursite.com).
    For each valid URL, provide a 1 sentence reasoning of why it fits the search query.
    Return ONLY valid JSON in this exact format:
    [
      {{
        "url": "https://real-website.com/path",
        "reasoning": "This site is highly relevant because..."
      }}
    ]
    
    Text: {raw_results}
    """
    
    json_output = _nova_lite(prompt)
    json_output = json_output.replace("```json", "").replace("```", "").strip()
    
    parsed_leads = []
    try:
        parsed_leads = json.loads(json_output)
    except json.JSONDecodeError:
        _log_to_db(db_session_id, "critic", "JSON parsing failed. Saving raw output.", queue)
        parsed_leads = [{"url": "Unknown", "reasoning": json_output}]
        
    for lead in parsed_leads:
        u = lead.get("url", "Unknown")
        r = lead.get("reasoning", "No reasoning provided.")
        _save_lead(db_session_id, u, r)
        _log_to_db(db_session_id, "system", f"Saved verified match: {u}", queue)
        
    return state

# Build Sequential Graph (No cycles)
def _build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("refiner", refiner_node)
    wf.add_node("executor", executor_node)
    wf.add_node("extractor", extractor_node)
    wf.set_entry_point("refiner")
    wf.add_edge("refiner", "executor")
    wf.add_edge("executor", "extractor")
    wf.add_edge("extractor", END)
    return wf.compile()

_GRAPH = _build_graph()

# ─── Main Execution Loop ─────────────────────────────────────────────────────

def run_lead_agent(db_session_id: int, goal: str, queue: asyncio.Queue | None = None):
    try:
        _mark_session(db_session_id, "running")
        _log_to_db(db_session_id, "system", f"Starting reasoning agent pipeline: '{goal}'", queue)

        initial_state: AgentState = {
            "db_session_id": db_session_id,
            "goal": goal,
            "refined_query": "",
            "raw_results": "",
            "log_queue": queue
        }
        _GRAPH.invoke(initial_state)
        
        _mark_session(db_session_id, "done")
        if queue:
            queue.put_nowait({"step": "system", "message": "DONE"})
            
    except Exception as e:
        print(f"--- NOVA ACT PIPELINE ERROR ---", file=sys.stderr)
        traceback.print_exc()
        print(f"-------------------------------", file=sys.stderr)
        
        _mark_session(db_session_id, "failed")
        msg = f"FAILED: {type(e).__name__} - {str(e)}"
        _log_to_db(db_session_id, "system", msg, queue)
        
        if queue:
            queue.put_nowait({"step": "system", "message": msg})


def run_outreach_agent(lead_id: int, url: str, name: str, email: str, description: str):
    """
    Runs Nova Act to automatically fill out a contact form on the given URL.
    """
    db = SessionLocal()
    try:
        lead = db.query(Lead).filter(Lead.id == lead_id).first()
        if lead:
            lead.contact_status = "running"
            db.commit()
            
            queue = get_or_create_queue(lead.session_id)
            _log_to_db(lead.session_id, "executor", f"Starting Outreach Agent for {url}", queue)
            
            boto_config = {"region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")}
            if os.environ.get("AWS_ACCESS_KEY_ID"):
                boto_config["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
                boto_config["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
                if os.environ.get("AWS_SESSION_TOKEN"):
                    boto_config["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")

            @workflow(
                workflow_definition_name="AutonomousMarketResearchAgent", 
                model_id="nova-act-latest",
                boto_session_kwargs=boto_config
            )
            def _execute_form_fill():
                with NovaAct(
                    starting_page=url,
                    headless=False,
                    tty=False
                ) as nova:
                    instruction = (
                        f"Navigate to the 'Contact Us' page on this website. "
                        f"Find the contact form. "
                        f"Fill the Name field with: '{name}'. "
                        f"Fill the Email field with: '{email}'. "
                        f"Fill the Message/Description field with: '{description}'. "
                        f"If there are any other required fields (like phone number or company), fill them with highly realistic dummy inputs. "
                        f"Click the Submit button. "
                        f"If you successfully submit the form, reply precisely with 'SUCCESS'. Otherwise explain the failure."
                    )
                    _log_to_db(lead.session_id, "executor", "Nova Act dispatched to fill contact form.", queue)
                    result = nova.act(instruction)
                    return str(result)

            try:
                result_str = _execute_form_fill()
                if "SUCCESS" in result_str.upper():
                    lead.contact_status = "success"
                    _log_to_db(lead.session_id, "system", f"Successfully contacted {url}", queue)
                else:
                    lead.contact_status = "failed"
                    _log_to_db(lead.session_id, "system", f"Failed to contact {url}: {result_str}", queue)
            except Exception as e:
                lead.contact_status = "failed"
                _log_to_db(lead.session_id, "system", f"Nova Act Form Fill Error: {str(e)}", queue)
                print(f"--- NOVA ACT OUTREACH ERROR ---", file=sys.stderr)
                traceback.print_exc()
                
            db.commit()
    finally:
        db.close()
