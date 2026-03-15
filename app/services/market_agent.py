import asyncio
import json
import os
import sys
import traceback
from typing import Optional, TypedDict
from datetime import datetime

import boto3
from langgraph.graph import StateGraph, END
from bs4 import BeautifulSoup
from nova_act import NovaAct, workflow

from app.db.database import SessionLocal
from app.models.market import MarketSession, MarketResult, MarketLog

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
    resp = client.invoke_model(modelId="global.amazon.nova-2-lite-v1:0", body=body)
    result = json.loads(resp["body"].read())
    return result["output"]["message"]["content"][0]["text"].strip()


# ─── DB & SSE Helpers ────────────────────────────────────────────────────────

def _log_to_db(db_session_id: int, step: str, message: str, queue=None):
    db = SessionLocal()
    try:
        log = MarketLog(session_id=db_session_id, step=step, message=message)
        db.add(log)
        db.commit()
    finally:
        db.close()
    if queue:
        try:
            queue.put_nowait({"step": step, "message": message})
        except asyncio.QueueFull:
            pass

def _save_result(db_session_id: int, url: str, dynamic_data: dict):
    db = SessionLocal()
    try:
        # Save as JSON string
        res = MarketResult(
            session_id=db_session_id,
            url=url,
            data=json.dumps(dynamic_data)
        )
        db.add(res)
        db.commit()
        db.refresh(res)
    finally:
        db.close()

def _mark_session(db_session_id: int, status: str):
    db = SessionLocal()
    try:
        s = db.query(MarketSession).get(db_session_id)
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
    goal: str # The user's idea pitch
    search_query: str
    competitor_urls: list[str]
    qualified_urls_map: dict[str, list[str]]
    raw_research_logs: list[dict] # {url: ..., raw_result: ...}
    log_queue: object
    fast_mode: bool # New flag for speed optimization

def planner_node(state: AgentState) -> AgentState:
    """Converts the startup idea into a competitor search query."""
    goal = state["goal"]
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    _log_to_db(db_session_id, "planner", f"Analyzing startup idea: '{goal}'", queue)
    
    prompt = f"""
    You are a market research strategist. Turn the following startup idea into a Google search query to find direct competitors or existing similar products.
    Do not add extra conversational text. Return ONLY the search string itself.
    Startup Idea: {goal}
    """
    search_query = _nova_lite(prompt).strip('"').strip("'")
    
    _log_to_db(db_session_id, "planner", f"Optimized search query: '{search_query}'", queue)
    state["search_query"] = search_query
    state["fast_mode"] = True # Default to fast mode
    return state

def executor_node(state: AgentState) -> AgentState:
    """Uses Tavily API to search Google for competitors with content extraction."""
    import httpx
    query = state["search_query"]
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    _log_to_db(db_session_id, "executor", f"Calling Tavily (FAST) to find competitors and extract content...", queue)
    
    tavily_key = os.environ.get("TAVILY_API_KEY")
    if not tavily_key:
        _log_to_db(db_session_id, "executor", "TAVILY_API_KEY not found.", queue)
        state["competitor_urls"] = []
        return state
        
    try:
        import time
        max_retries = 3
        data = None
        
        with httpx.Client(timeout=30.0) as client:
            for attempt in range(max_retries):
                try:
                    response = client.post(
                        "https://api.tavily.com/search",
                        json={
                            "api_key": tavily_key,
                            "query": query,
                            "search_depth": "advanced", # Advanced for better content
                            "include_content": True,     # GET CONTENT DIRECTLY
                            "max_results": 4
                        }
                    )
                    response.raise_for_status()
                    data = response.json()
                    break # Success
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(2 ** attempt)
        
        results = data.get("results", [])
        urls = []
        for r in results:
            url = r.get('url')
            content = r.get('content')
            if url:
                urls.append(url)
                if content:
                    # Store the extracted content directly into raw_research_logs
                    state["raw_research_logs"].append({
                        "url": url,
                        "raw_result": f"Content extracted from Tavily Search:\n{content}"
                    })
        
        _log_to_db(db_session_id, "executor", f"Tavily returned {len(urls)} competitors with instant content.", queue)
        state["competitor_urls"] = urls
    except Exception as e:
        _log_to_db(db_session_id, "executor", f"Tavily Error: {str(e)}", queue)
        state["competitor_urls"] = []
        
    return state

def filter_node(state: AgentState) -> AgentState:
    """Filters out irrelevant URLs and decides if we can skip deep scraping."""
    urls = state.get("competitor_urls", [])
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    _log_to_db(db_session_id, "critic", f"Filtering {len(urls)} discovered URLs...", queue)
    
    filtered_urls = []
    ignore_domains = ["wikipedia.org", "youtube.com", "facebook.com", "reddit.com", "ycombinator.com", "forbes.com", "techcrunch.com"]
    
    for u in urls:
        if any(domain in u.lower() for domain in ignore_domains):
            _log_to_db(db_session_id, "critic", f"Discarded non-product URL: {u}", queue)
        else:
            filtered_urls.append(u)
            
    state["competitor_urls"] = filtered_urls[:4]
    
    # If we already have content for these URLs, we can set fast_mode = True to jump to extraction
    has_logs = {log["url"] for log in state["raw_research_logs"]}
    if all(u in has_logs for u in state["competitor_urls"]) and state["competitor_urls"]:
        _log_to_db(db_session_id, "system", "Content available via quick search. Skipping deep browse.", queue)
        state["fast_mode"] = True
    else:
        state["fast_mode"] = False # Need deep scraping if content is missing
        
    return state

def discovery_node(state: AgentState) -> AgentState:
    """Uses Nova Act to visit the homepage and explicitly extract all internal links."""
    if state.get("fast_mode"):
        return state # Skip in fast mode

    urls = state.get("competitor_urls", [])
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    goal = state["goal"]
    qualified_urls_map = {}

    if not urls:
        _log_to_db(db_session_id, "critic", "No valid URLs to discover.", queue)
        state["qualified_urls_map"] = {}
        return state

    boto_config = {"region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")}
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        boto_config["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
        boto_config["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if os.environ.get("AWS_SESSION_TOKEN"):
            boto_config["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")

    for url in urls:
        _log_to_db(db_session_id, "executor", f"Discovering internal links for {url} using Nova Act", queue)
        try:
            @workflow(workflow_definition_name="AutonomousMarketResearchAgent", model_id="nova-act-latest", boto_session_kwargs=boto_config)
            def _get_links():
                with NovaAct(starting_page=url, headless=True) as nova:
                    instruction = "Find and return all unique internal URLs on this page that lead to pricing, features, about, or product details. Return them as a comma-separated list."
                    res = nova.act(instruction)
                    return str(res)
            
            links_raw = _get_links()
            links_list = [l.strip() for l in links_raw.split(',') if l.strip().startswith('http')]
            
            if not links_list:
                import httpx
                from bs4 import BeautifulSoup
                from urllib.parse import urljoin, urlparse
                with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                    resp = client.get(url)
                    soup = BeautifulSoup(resp.content, "html.parser")
                    base_parsed = urlparse(resp.url)
                    base_url = f"{base_parsed.scheme}://{base_parsed.netloc}"
                    internal_links = set()
                    for a_tag in soup.find_all("a", href=True):
                        href = a_tag["href"]
                        full_url = urljoin(base_url, href)
                        if full_url.startswith(base_url):
                            clean_url = full_url.split("#")[0].split("?")[0]
                            if clean_url != base_url:
                                internal_links.add(clean_url)
                    links_list = list(internal_links)

            if len(links_list) > 30:
                links_list = links_list[:30]

            if links_list:
                prompt = f"You are a market research qualifier.\nHere are links from a competitor site ({url}).\nResearch Goal: '{goal}'.\nSelect up to 4 links most likely to have pricing, features, or product details.\nReturn JUST the URLs, one per line. No markdown.\n\nLinks:\n" + "\n".join(links_list)
                top_links_raw = _nova_lite(prompt)
                top_links = [l.strip() for l in top_links_raw.split('\n') if l.strip() and (l.strip() in links_list or l.strip().startswith('http'))]
                top_links = top_links[:4]
            else:
                top_links = []
            
            all_to_scrape = [url] + top_links
            seen = set()
            final_list = [x for x in all_to_scrape if not (x in seen or seen.add(x))]
            
            qualified_urls_map[url] = final_list
            _log_to_db(db_session_id, "critic", f"Found {len(top_links)} targets for {url}", queue)
        except Exception as e:
            _log_to_db(db_session_id, "critic", f"Discovery failed for {url}: {e}", queue)
            qualified_urls_map[url] = [url]
            
    state["qualified_urls_map"] = qualified_urls_map
    return state

def nova_scraper_node(state: AgentState) -> AgentState:
    """Uses Nova Act to visit the qualified URLs and extract text."""
    if state.get("fast_mode"):
        return state # Skip in fast mode

    qualified_map = state.get("qualified_urls_map", {})
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    
    if not qualified_map:
        return state
        
    boto_config = {"region_name": os.environ.get("AWS_DEFAULT_REGION", "us-east-1")}
    if os.environ.get("AWS_ACCESS_KEY_ID"):
        boto_config["aws_access_key_id"] = os.environ.get("AWS_ACCESS_KEY_ID")
        boto_config["aws_secret_access_key"] = os.environ.get("AWS_SECRET_ACCESS_KEY")
        if os.environ.get("AWS_SESSION_TOKEN"):
            boto_config["aws_session_token"] = os.environ.get("AWS_SESSION_TOKEN")
            
    for main_url, urls_to_scrape in qualified_map.items():
        combined_text = ""
        _log_to_db(db_session_id, "executor", f"Deep scraping {len(urls_to_scrape)} pages for {main_url} using Nova Act", queue)
        
        for url in urls_to_scrape:
            @workflow(workflow_definition_name="AutonomousMarketResearchAgent", model_id="nova-act-latest", boto_session_kwargs=boto_config)
            def _scrape_page():
                with NovaAct(starting_page=url, headless=True) as nova:
                    instruction = "Extract all the text content visible on this page. Just dump the text directly."
                    res = nova.act(instruction)
                    return str(res)
            
            try:
                page_text = _scrape_page()
                combined_text += f"\n\n--- Content from {url} ---\n{page_text}"
            except Exception as e:
                _log_to_db(db_session_id, "critic", f"Failed to scrape {url} with Nova Act. Error: {e}", queue)
                
        if combined_text:
            state["raw_research_logs"].append({"url": main_url, "raw_result": combined_text[:40000]})
            _log_to_db(db_session_id, "system", f"Successfully gathered data for {main_url}", queue)
        else:
            _log_to_db(db_session_id, "critic", f"No text data retrieved for {main_url}", queue)
            
    return state

def extractor_node(state: AgentState) -> AgentState:
    """Takes the raw content and distills only the website intelligence into JSON."""
    logs = state.get("raw_research_logs", [])
    queue = state.get("log_queue")
    db_session_id = state["db_session_id"]
    goal = state["goal"]

    _log_to_db(db_session_id, "critic", "Distilling rich insights from gathered content...", queue)

    for item in logs:
        url = item["url"]
        raw_text = item["raw_result"]

        prompt = f"""
        You are a data extractor. Extract business intelligence from the following content regarding this product.
        
        Research Goal: '{goal}'.
        
        Generate dynamic Key-Value pairs. 
        Ensure you ALWAYS include:
        - "Pricing": Detailed pricing or 'Contact for pricing' if not found.
        - "Key Capabilities": Bullet points of main features.
        - "Idea Relevance": Comparison to the user's idea.

        Return ONLY a raw, flat JSON dictionary. No markdown, no pre-text.
        
        Content:
        {raw_text[:15000]}
        """

        try:
            json_output = _nova_lite(prompt)
            if "```json" in json_output:
                json_output = json_output.split("```json")[1].split("```")[0]
            elif "```" in json_output:
                json_output = json_output.split("```")[1].split("```")[0]
                
            parsed = json.loads(json_output.strip())
            _save_result(db_session_id, url, parsed)
            _log_to_db(db_session_id, "critic", f"Structured intelligence for {url}", queue)
        except Exception as e:
            _log_to_db(db_session_id, "critic", f"Failed to structure data for {url}. Saving raw snippet.", queue)
            _save_result(db_session_id, url, {"Error": "Extraction Failed", "Raw Data Snippet": raw_text[:500] + "..."})

    return state


def _build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("planner", planner_node)
    wf.add_node("executor", executor_node)
    wf.add_node("filter", filter_node)
    wf.add_node("discovery", discovery_node)
    wf.add_node("nova_scraper", nova_scraper_node)
    wf.add_node("extractor", extractor_node)
    
    wf.set_entry_point("planner")
    wf.add_edge("planner", "executor")
    wf.add_edge("executor", "filter")
    wf.add_edge("filter", "discovery")
    wf.add_edge("discovery", "nova_scraper")
    wf.add_edge("nova_scraper", "extractor")
    wf.add_edge("extractor", END)
    return wf.compile()

_GRAPH = _build_graph()

# ─── Main Execution Loop ─────────────────────────────────────────────────────

def run_market_agent(db_session_id: int, goal: str, queue: asyncio.Queue | None = None):
    try:
        _mark_session(db_session_id, "running")
        _log_to_db(db_session_id, "system", f"Starting Nova Quick Search: '{goal}'", queue)

        initial_state: AgentState = {
            "db_session_id": db_session_id,
            "goal": goal,
            "search_query": "",
            "competitor_urls": [],
            "qualified_urls_map": {},
            "raw_research_logs": [],
            "log_queue": queue,
            "fast_mode": True
        }
        _GRAPH.invoke(initial_state)
        
        _mark_session(db_session_id, "done")
        if queue:
            queue.put_nowait({"step": "system", "message": "DONE"})
            
    except Exception as e:
        print(f"--- MARKET RESEARCH PIPELINE ERROR ---", file=sys.stderr)
        traceback.print_exc()
        print(f"-------------------------------", file=sys.stderr)
        
        _mark_session(db_session_id, "failed")
        msg = f"FAILED: {type(e).__name__} - {str(e)}"
        _log_to_db(db_session_id, "system", msg, queue)
        
        if queue:
            queue.put_nowait({"step": "system", "message": msg})
