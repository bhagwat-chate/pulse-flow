# PulseFlow – Multi-Agent Product Intelligence System
> FAANG-grade, production-ready multi-agent RAG framework with LangGraph + MCP + RAGAs + LangSmith observability

---

### Overview
PulseFlow is an **agentic AI orchestration framework** built for **e-commerce product intelligence** and **domain-adaptive reasoning**.  
It blends **Retrieval-Augmented Generation (RAG)**, **LangGraph orchestration**, **Multi-Channel Processing (MCP)** for hybrid tool use, and **RAGAs-based evaluation** for precision tracking.

### Core features:
 🔹 **Multi-Agent Architecture** — Router → Retriever → Grader → Rewriter → Generator  
 🔹 **Hybrid Retrieval** — Combines vector recall (AstraDB) + real-time web search (DuckDuckGo via MCP)  
 🔹 **Async Evaluation** — Context Precision & Response Relevancy (RAGAs)  
 🔹 **Structured Observability** — Full JSON logging + trace correlation via StructLog & LangSmith  
 🔹 **Cloud-Native Ready** — Dockerized, Kubernetes deployable, AWS Secrets-integrated  

---

### 🏗️ Architecture Snapshot
User → Router → Retriever → Grader → Rewriter → WebSearch → Generator → RAGAs Eval
│
├── MCP Adapter → AstraDB (Vector) + DuckDuckGo (Web)
└── LangGraph StateGraph (Checkpointed via MemorySaver)


---

### 🧩 Technology Stack

| Layer | Tools / Frameworks | Purpose |
|-------|--------------------|----------|
| **LLM & Prompting** | OpenAI GPT-4o, Groq Mixtral, LangChain | Core reasoning & generation |
| **Orchestration** | LangGraph 0.6 (StateGraph + MemorySaver) | Multi-agent flow control |
| **Retrieval** | AstraDB Vector Store + Contextual Compression | Semantic recall |
| **Tooling** | MCP (FastMCP + Adapters) | Asynchronous external tool execution |
| **Evaluation** | RAGAs + LangSmith Tracing | Response & context metrics |
| **Infra** | Docker, EKS, ECR, AWS Secrets Manager | Cloud deployment |
| **Logging** | StructLog + Trace IDs + JSON logs | Enterprise observability |

---

### ⚙️ Local Setup

```bash
# 1️⃣ Clone the repo
git clone https://github.com/bhagwat-chate/pulse-flow.git
cd pulse-flow

# 2️⃣ Create virtual environment
uv venv venv-pulse-flow
source venv-pulse-flow/bin/activate  # or .\venv-pulse-flow\Scripts\activate

# 3️⃣ Install dependencies
pip install -r requirements.txt

# 4️⃣ Configure environment
cp .env.example .env
# → Fill in API keys & AstraDB info for local dev

# 5️⃣ Run FastAPI service
python main.py

Visit → http://localhost:8080
You’ll see the interactive PulseFlow chat UI powered by templates/chat.html.
```
### AWS / Kubernetes Deployment
1️⃣ Build & Push Docker Image
```commandline
docker build -t pulseflow:latest .
docker tag pulseflow:latest <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/pulseflow:latest
docker push <aws_account_id>.dkr.ecr.ap-south-1.amazonaws.com/pulseflow:latest
```

2️⃣ Apply Kubernetes Manifests
```commandline
kubectl apply -f k8/deployment.yaml
kubectl apply -f k8/service.yaml
```
3️⃣ Secrets Management

All production secrets are loaded from AWS Secrets Manager
(AWS_SECRET_NAME + AWS_REGION in container env).

core/bootstrap.py merges them automatically at startup.

### Repository Layout (v1.0.0)
```
prod_assistant/
├── core/            # bootstrap, logging, tracing
├── utils/           # shared loaders (model, MCP, RAGAs)
├── workflow/        # LangGraph multi-agent pipelines
├── mcp_servers/     # external MCP tools (product_search_server)
├── retriever/       # AstraDB retriever
├── router/          # FastAPI app, routes
├── prompt_library/  # prompt registry
├── evaluation/      # RAGAs evaluation logic
└── exception/       # domain-level exception classes
```
### Example Query Flow
- User asks → “iPhone 15 Plus price in India” 
- Router decides retrieval path → MCP get_product_info
- Retriever fetches from AstraDB or DuckDuckGo
- Grader validates relevance → rewrites if unclear
- Generator composes final answer
- RAGAs evaluates → logs context precision + response relevancy

Logs per node are trace-correlated (trace_id) and streamed to LangSmith.

### RAG Evaluation Metrics
| Metric                 | Description                                                |
| ---------------------- | ---------------------------------------------------------- |
| **Context Precision**  | Measures overlap between retrieved vs ground truth context |
| **Response Relevancy** | Evaluates LLM answer alignment with input intent           |
| **Composite Score**    | Weighted average for continuous benchmarking               |

### Deployment Checkpoints
| Environment | Config File               | Notes                           |
| ----------- | ------------------------- | ------------------------------- |
| **DEV**     | `config/config_dev.yaml`  | Loads from `.env`, verbose logs |
| **PROD**    | `config/config_prod.yaml` | AWS Secrets + minimal logs      |
| **Base**    | `config/config_base.yaml` | Shared defaults                 |

### Core MCP Tools
| Tool Name          | Purpose                            | Backend               |
| ------------------ | ---------------------------------- | --------------------- |
| `get_product_info` | Product retrieval from AstraDB     | LangChain Retriever   |
| `web_search`       | Web query fallback (news, reviews) | DuckDuckGo Search Run |

### Observability

- Structured JSON logs via StructLog
- Trace IDs per HTTP request (trace_middleware.py)
- LangSmith integration for node-level traces
- Logs can be exported to CloudWatch / Loki / ELK

### Development Notes
- Code adheres to PEP 8 + FAANGM-grade documentation standards
- Exception safety across all nodes (custom_exception.py)
- Fully asynchronous MCP tool handling (see mcp_tool_loader.py)
- Supports domain adaptation (finance, healthcare, retail etc.)

### Versioning
| Property     | Value         |
| ------------ | ------------- |
| Version      | `1.0.0`       |
| Release Date | `2025-10-10`  |
| Maintainer   | Bhagwat Chate |
| License      | MIT           |
| Python       | 3.11.13       |


### Credits
Developed by **Bhagwat Chate**
→ GenAI Architect | System Design Mentor | LLMOPS Consultant

LinkedIn: AI with Bhagwat Chate

GitHub: bhagwat-chate/pulse-flow

### License
This project is licensed under the MIT License.
See the full license text in LICENSE.
