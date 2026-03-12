# Nova AEI System Flows

This document maps the specific code locations and logic flows for the system's core AI features.

## 1. Facial Recognition & Surveillance Flow
The system streams video and processes it using **Nova 2 Lite** combined with local vector matching.

**Files:** [manager.py](file:///c:/Users/venka/Nova%20AEI/app/manager.py), [face_service.py](file:///c:/Users/venka/Nova%20AEI/app/services/face_service.py), [chromadb_service.py](file:///c:/Users/venka/Nova%20AEI/app/services/chromadb_service.py)

1.  **S3 Upload**: `SurveillanceManager` streams and uploads mp4 video chunks every 30 seconds to S3.
2.  **Vision Analysis**: **Nova 2 Lite** evaluates the video directly from S3 to detect surveillance events or check if the person model exists.
3.  **Local Face Match**: If an event triggers, **OpenCV** extracts faces and queries **ChromaDB** for person identification to verify if they are authorized.

## 2. Voice Assistant & Knowledge Base Flow
Handles real-time "Reasoning" over company documents, product queries, and support tickets.

**Files:** [bedrock_streaming.py](file:///c:/Users/venka/Nova%20AEI/app/services/bedrock_streaming.py), [agent.py](file:///c:/Users/venka/Nova%20AEI/app/routers/agent.py)

1.  **Real-time Streaming**: Uses **Nova 2 Sonic** via bidirectional WebSocket streaming for low-latency voice interaction.
2.  **Knowledge Base (Multimodal Context)**: When a user query matches technical/product issues, the system triggers `query_knowledge_base`. 
3.  **Embeddings Retrieval**: The **AWS Bedrock Knowledge Base** utilizes **Nova Multimodal Embeddings** to retrieve relevant context from indexed **PDFs, text, and images**, which is then injected into the Nova 2 Sonic conversation for "reasoned" responses.
4.  **Ticket Generation**: If the KB doesn't resolve the issue, the assistant automatically calls `raise_ticket_in_db` to log a support ticket.

## 3. Market Research & Vendor Search Flow
Uses autonomous browser actions to research competitors and find the best vendors.

**Files:** [market_agent.py](file:///c:/Users/venka/Nova%20AEI/app/services/market_agent.py), [lead_agent.py](file:///c:/Users/venka/Nova%20AEI/app/services/lead_agent.py)

1.  **Market Research**: Uses **Nova Act** combined with the **Tavily Search API** to identify current market competitors based on our product ideas and scrape their information.
2.  **Vendor Search**: The system searches for vendors, specifying requirements. It uses **Nova Act** + **Tavily** to find vendors providing the lowest prices, and automatically fills out their contact forms.

## 4. DevOps Agents (Functions) Flow
Streams real-time repository management, code generation, and cost optimization.

**File:** [functions.py](file:///c:/Users/venka/Nova%20AEI/app/routers/functions.py)

1.  **Invoke**: Agent starts via `/functions/invoke`. Built using **Nova Pro** and **Nova Premier** models.
2.  **Task Execution (Lambda)**: The agents (Code Generation, Code Scan, Cost Optimization) use Bedrock Agents to take repository URLs, write/fix code directly, and push the code using the Lambda function action groups.
