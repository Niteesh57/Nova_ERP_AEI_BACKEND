# Nova AEI Service & Process Overview

This document provides a detailed breakdown of the services and AI models used across every feature of the Nova AEI system, along with the specific processes followed.

---

## 1. Surveillance & Real-Time Streaming
Handles automated monitoring and intrusion detection via video analysis and facial recognition.

### Services & Models
| Category | Service / Model | Purpose |
| :--- | :--- | :--- |
| **Storage** | **Amazon S3** | Stores 30-second mp4 video chunks for model processing. |
| **Vision AI** | **Amazon Nova 2 Lite** | Analyzes video files directly from S3 to detect specific events. |
| **Computer Vision**| **OpenCV** | Extracts facial frames from video chunks for local processing. |
| **Vector DB** | **ChromaDB** | Stores and matches face embeddings for person identification. |
| **Identification** | **Bedrock Embeddings**| Generates high-quality vector representations of faces. |

### Process Flow
1.  **Capture**: `SurveillanceManager` receives `webm` video chunks (30s intervals).
2.  **Transcode**: Chunks are transcoded to `mp4` for AWS Bedrock compliance.
3.  **Upload**: The `mp4` file is uploaded to **Amazon S3**.
4.  **Nova Analysis**: **Nova 2 Lite** is invoked with the S3 URI to detect configured events (e.g., "Person detected").
5.  **Face Extraction**: If an event triggers, **OpenCV** extracts faces from the video chunk.
6.  **ChromaDB Match**: Extracted faces are matched against **ChromaDB** to identify authorized employees.
7.  **Alerting**: If an unauthorized person is detected, an alert is generated and sent to the administrator.

---

## 2. Voice Assistant & Knowledge Reasoning
Provides real-time, "reasoned" voice interaction based on company-specific data.

### Services & Models
| Category | Service / Model | Purpose |
| :--- | :--- | :--- |
| **Real-time Voice** | **Amazon Nova 2 Sonic** | Handles bidirectional WebSocket streaming for low-latency dialogue. |
| **Embeddings** | **Nova Multimodal Embeddings** | Powers the search across text, PDFs, and images. |
| **Knowledge Base** | **AWS Bedrock KB** | Conducts RAG (Retrieval-Augmented Generation) over indexed docs. |
| **Persistence** | **SQLite** | Logs all conversations and support tickets. |

### Process Flow
1.  **WebSocket Sync**: Establishes a bidirectional stream using **Nova 2 Sonic**.
2.  **Multimodal Search**: When queries involve technical data, the system queries the **Bedrock Knowledge Base**.
3.  **RAG Enrichment**: Relevant snippets (from PDFs/Images) are retrieved using **Nova Multimodal Embeddings**.
4.  **Inference**: **Nova 2 Sonic** generates a response enriched by the retrieved context.
5.  **Ticketing**: If the query remains unresolved, the system automatically uses a tool to `raise_ticket_in_db`.

---

## 3. Market Intelligence & Vendor Search
Autonomous agents that perform deep-web research and vendor outreach.

### Services & Models
| Category | Service / Model | Purpose |
| :--- | :--- | :--- |
| **Autonomous Web** | **Nova Act** | Operates a headless browser to discovery links and scrape deep content. |
| **Search API** | **Tavily Search** | Provides high-quality, real-time web search results. |
| **Strategic Logic** | **Amazon Nova Lite** | Converts ideas into search queries and distills scraped data into insights. |
| **Outreach** | **Nova Act** | Automatically navigates to contact forms and fills them out. |

### Process Flow
1.  **Planner**: **Nova Lite** turns a product pitch into an optimized search query.
2.  **Discovery**: **Tavily Search** returns high-level competitor/vendor URLs.
3.  **Deep Scraping**: **Nova Act** navigates to the found URLs, identifies pricing/feature pages, and scrapes full text.
4.  **Intel Extraction**: **Nova Lite** parses noisy web data into a structured JSON of "Dynamic Intelligence" (Pricing, Capabilities, etc.).
5.  **Vendor Outreach**: For vendor leads, **Nova Act** navigates to "Contact Us" pages and submits the specified requirements.

---

## 4. DevOps Functions (AI Code Agents)
Streams real-time repository management, code generation, and cost optimization.

### Services & Models
| Category | Service / Model | Purpose |
| :--- | :--- | :--- |
| **Agent Orchestration**| **AWS Bedrock Agents** | Manages specialized sessions for Coding, Scanning, and Optimization. |
| **Main LLMs** | **Nova Pro / Premier** | Powers the high-reasoning code generation and scanning tasks. |
| **Action Groups** | **AWS Lambda** | Executes repo-level commands (Git push, file edits). |
| **Streaming** | **SSE (Server-Sent Events)** | Provides real-time progress updates to the frontend. |

### Process Flow
1.  **Invoke**: Frontend selects an agent type (`code_generation`, `code_scanner`, or `cost_optimizer`).
2.  **Execution**: The **Bedrock Agent** (using **Nova Pro/Premier**) analyzes the repository via a **Lambda** function.
3.  **Human-in-the-Loop**: If an agent needs to perform a sensitive action (like pushing code), it emits a `returnControl` event.
4.  **Confirmation**: The user confirms/denies the action via the UI, and the agent resumes execution.
5.  **Completion**: The final output/status is streamed back to the user via SSE.
