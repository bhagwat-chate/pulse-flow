# Pulse-Flow

**Smart E-Commerce Assistant with Real-Time Product Intelligence**  

Pulse-Flow is a **GenAI-powered product intelligence system** that can analyze and respond to real-time product queries by merging **static product data** with **live e-commerce information**. Built with **RAG + LangChain agents + Multi-Channel Processing (MCP)**, it delivers accurate, context-aware, multi-step reasoning for e-commerce applications.  

---

## 🚀 Project Vision  

In today’s fast-moving e-commerce world, customers demand **instant, intelligent answers** to product queries. Traditional systems rely only on static catalogs, missing out on **real-time signals** like discounts, availability, or reviews.  

**Pulse-Flow bridges this gap** — enabling enterprises to deliver **personalized, reliable, and real-time product insights** at scale.  

---

## 🏗️ System Architecture  

comming soon...


---

## 🔑 Core Features  

✨ **RAG + LangChain Agents**  
Ground responses in factual product knowledge while enabling **multi-step reasoning** (e.g., “Compare top 3 budget smartphones under ₹25K with best battery life”).  

✨ **Multi-Channel Processing (MCP)**  
Merge static + dynamic signals to deliver context-aware answers, ensuring responses stay fresh and market-aligned.  

✨ **Real-Time Data Enablement**  
Use **APIs & scraping pipelines** to fetch product details, reviews, and live availability from platforms like Amazon & Flipkart.  

✨ **Decoupled Frontend & Backend**  
- **Frontend**: Chat UI built with HTML/CSS/JS.  
- **Backend**: FastAPI-powered APIs for modular query processing.  

✨ **CI/CD + Secure Deployment**  
- GitHub Actions for automated builds & rollouts.  
- Docker & Trivy for vulnerability scanning.  
- AWS EKS for secure, scalable Kubernetes deployments.  

---

## 📚 Learning Tracks  

1️⃣ **Foundation of Product Intelligence Systems**  
- Introduction to Intelligent Product Assistants.  
- RAG and MM-RAG overview.  

2️⃣ **Real-Time Data Pipelines**  
- APIs & scraping workflows.  
- Preparing structured product datasets.  
- Embedding & indexing with Astra DB.  

3️⃣ **Merged Context AI with MCP**  
- Building RAG Chains for blended context.  
- LangChain tools & agents for multi-step queries.  
- Role of MCP in enterprise architectures.  

4️⃣ **Frontend & Backend Development**  
- Chat UI for real-time interactions.  
- FastAPI backend for scalable query pipelines.  

5️⃣ **CI/CD & Secure Deployment**  
- GitHub Actions automation.  
- Docker image build & push to AWS ECR.  
- Trivy security scans.  
- Kubernetes manifests (`deployment.yaml`, `service.yaml`, `configmap.yaml`).  
- AWS EKS rollout verification & monitoring.  

---

## 🛠️ Tech Stack  

- **GenAI Layer**: RAG, LangChain, MCP  
- **Data**: Astra DB, APIs, Web Scraping  
- **Backend**: FastAPI  
- **Frontend**: HTML, CSS, JS (Chat UI)  
- **DevOps**: GitHub Actions, Docker, Trivy, AWS ECR  
- **Deployment**: Kubernetes, AWS EKS  

---

## 📦 Repository Structure  

```bash
pulse-flow/
├── .github
│   └── workflows
│       └── .gitkeep
├── .gitignore
├── LICENSE
├── README.md
├── get_project_structure.py
├── main.py
├── notebook
│   └── experiment.ipynb
├── prod_assistant
│   ├── config
│   │   └── config.yaml
│   ├── etl
│   │   ├── data_ingestion.py
│   │   └── data_scrapper.py
│   ├── exception
│   │   └── custom_exception.py
│   ├── logger
│   │   └── custom_logger.py
│   └── utils
│       ├── api_key_manager.py
│       ├── config_loader.py
│       └── model_loader.py
├── pyproject.toml
├── requirements.txt
├── scrapper_ui.py
├── static
│   └── online_shopping.png
├── templates
│   └── chat.html
└── versions.py

```

---

## ✅ Deployment Flow  

1. **Code Push → GitHub**  
2. **GitHub Actions CI/CD**  
   - Build Docker image.  
   - Run Trivy vulnerability scans.  
   - Push to AWS ECR.  
3. **GitHub Actions CD**  
   - Deploy manifests to AWS EKS.  
   - Verify rollout & health checks.  
4. **Monitoring**  
   - Logs & metrics via CloudWatch.  

---

## 🎯 Outcomes  

By the end of this project, you’ll have:  
- A **production-ready GenAI e-commerce assistant**.  
- Hands-on expertise in **RAG, LangChain, and MCP architectures**.  
- Skills in **secure DevSecOps pipelines with AWS EKS**.  
- A **portfolio-ready, enterprise-grade project** to showcase.  

---

## 🔗 Connect  

💡 Built as part of my journey into **GenAI system design & enterprise deployments**.  
Follow my LinkedIn series for deep-dives, lessons learned, and live progress updates on **Pulse-Flow**.  
