# prod_assistant/workflow/agentic_workflow_with_mcp_websearch.py

import json
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing import Literal
from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
import asyncio
from prod_assistant.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient


class AgenticRAG:
    """Agentic RAG pipeline using LangGraph + MCP (Retriever + WebSearch)."""

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()

        # MCP Client Init
        self.mcp_client = MultiServerMCPClient({
                    "hybrid_search": {
                        "command": "python",
                        "args": ["-m", "prod_assistant.mcp_servers.product_search_server"],
                        "transport": "stdio"
                    }
                })
        # Load MCP tools
        # self.mcp_tools = asyncio.run(self.mcp_client.get_tools())
        try:
            self.mcp_tools = asyncio.run(self.mcp_client.get_tools())
            print(f"[MCP] Loaded tools: {[t.name for t in self.mcp_tools]}")

        except Exception as e:
            print(f"[MCP] Failed to load tools: {e}")
            self.mcp_tools = []

        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ---------- Nodes ----------
    # def _ai_assistant(self, state: AgentState):
    #     print("--- CALL ASSISTANT ---")
    #     messages = state["messages"]
    #     last_message = messages[-1].content
    #
    #     if any(word in last_message.lower() for word in ["price", "review", "product"]):
    #         return {"messages": [HumanMessage(content="TOOL: retriever")]}
    #     else:
    #         prompt = ChatPromptTemplate.from_template(
    #             "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
    #         )
    #         chain = prompt | self.llm | StrOutputParser()
    #         response = chain.invoke({"question": last_message})
    #         return {"messages": [HumanMessage(content=response)]}

    def _ai_assistant(self, state: AgentState):
        print("--- CALL ASSISTANT ---")
        RouteDecision = Literal["retriever", "web_search", "direct"]

        last_message = state["messages"][-1].content

        router_prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.ROUTER_BOT].template
        )
        chain = router_prompt | self.llm | StrOutputParser()
        decision_str = chain.invoke({"query": last_message}).strip().lower()

        # Normalize quotes/spaces
        decision_str = decision_str.replace('"', '').replace("'", "").strip()

        decision: RouteDecision
        if decision_str in ("retriever", "web_search", "direct"):
            decision = decision_str  # type: ignore
        else:
            print(f"[ROUTER ERROR] Invalid decision: {decision_str}")
            decision = "direct"

        print(f"[ROUTER DECISION] {decision}")

        if decision == "retriever":
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        elif decision == "web_search":
            return {"messages": [HumanMessage(content="TOOL: web_search")]}
        else:
            # Direct answer flow
            answer_prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = answer_prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("--- RETRIEVER (MCP) ---")
        query = state["messages"][-1].content

        tool = next((t for t in self.mcp_tools if t.name == "get_product_info"), None)
        if not tool:
            return {"messages": [HumanMessage(content="[MCP ERROR] get_product_info tool not available")]}

        result = asyncio.run(tool.ainvoke({"query": query}))
        print(f"[DEBUG] Raw result from MCP: {result} (type: {type(result)})")

        # Always try JSON parse if string
        docs = []
        if isinstance(result, str):
            try:
                docs = json.loads(result)  # now [] becomes proper empty list
            except Exception as e:
                print(f"[DEBUG] JSON parse failed: {e}")
                docs = []
        elif isinstance(result, list):
            docs = result
        else:
            print(f"[DEBUG] Unexpected type from MCP: {type(result)}")

        # Format docs
        if not docs:
            context = "No local results found."
        else:
            print(f"total docs retrieved: {len(docs)}")
            context = "\n\n---\n\n".join([
                f"Title: {doc.get('metadata', {}).get('product_title', 'N/A')}\n"
                f"Price: {doc.get('metadata', {}).get('price', 'N/A')}\n"
                f"Rating: {doc.get('metadata', {}).get('rating', 'N/A')}\n"
                f"Reviews:\n{doc.get('page_content', '')}"
                for doc in docs
            ])

        return {"messages": [HumanMessage(content=context)]}

    def _web_search(self, state: AgentState):
        print("--- WEB SEARCH (MCP) ---")
        query = state["messages"][-1].content

        # tool = next(t for t in self.mcp_tools if t.name == "web_search")
        tool = next((t for t in self.mcp_tools if t.name == "web_search"), None)
        if not tool:
            return {"messages": [HumanMessage(content="[MCP ERROR] web_search tool not available")]}

        result = asyncio.run(tool.ainvoke({"query": query}))
        context = result if result else "No data from web"
        return {"messages": [HumanMessage(content=context)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("--- GENERATE ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})
        return {"messages": [HumanMessage(content=response)]}

    def _rewrite(self, state: AgentState):
        print("--- REWRITE ---")
        question = state["messages"][0].content
        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine. "
            "Do NOT answer the query. Only rewrite it.\n\nQuery: {question}\nRewritten Query:"
        )
        chain = prompt | self.llm | StrOutputParser()
        new_q = chain.invoke({"question": question})
        return {"messages": [HumanMessage(content=new_q.strip())]}

    # ---------- Build Workflow ----------
    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)
        workflow.add_node("Assistant", self._ai_assistant)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)
        workflow.add_node("WebSearch", self._web_search)

        workflow.add_edge(START, "Assistant")
        workflow.add_conditional_edges(
            "Assistant",

            lambda state: "Retriever" if "TOOL" in state["messages"][-1].content else END,

            {
                "Retriever": "Retriever",
                END: END
            },
        )
        workflow.add_conditional_edges(

            "Retriever",

            self._grade_documents,

            {"generator": "Generator",

             "rewriter": "Rewriter"},
        )
        workflow.add_edge("Generator", END)

        workflow.add_edge("Rewriter", "WebSearch")

        workflow.add_edge("WebSearch", "Generator")

        return workflow

    # ---------- Public Run ----------
    def run(self, query: str, thread_id: str = "default_thread") -> str:
        """Run the workflow for a given query and return the final answer."""
        result = self.app.invoke({"messages": [HumanMessage(content=query)]},
                                 config={"configurable": {"thread_id": thread_id}})
        return result["messages"][-1].content


if __name__ == "__main__":
    # query = "What do customers say about the battery life of the iPhone 15 Plus?"
    # query = "List two negative reviews regarding the iPhone 15 Plus performance."
    # query = "What is the price of iPhone 15 Plus in India?"
    # query = "Summarize top customer opinions about the Samsung S24’s battery life."
    # query = "List three negative reviews for iPhone 15 Plus."
    # print(f"[query] query: {query}")
    # rag_agent = AgenticRAG()
    # print(f"[query] {query}")
    # answer = rag_agent.run(query)
    # print("\nFinal Answer:\n", answer)


    # test_queries = [
    #     "What do customers say about the battery life of the iPhone 15 Plus?",
    #     "Summarize three positive opinions about the iPhone 15 camera quality.",
    #     "How do buyers describe the display brightness and clarity of the iPhone 15?",
    #     "Do users feel that the iPhone 15 is worth the price they paid?",
    #     "Summarize customer feedback about the design and build quality of the iPhone 15 Plus.",
    #     "What are the most common complaints about the iPhone 15 heating issues?",
    #     "What positive experiences do customers report about the iPhone 15 Plus battery charging speed?",
    #     "Summarize how customers reviewed the speaker and sound quality of the iPhone 15.",
    #     "Compare customer opinions on durability between iPhone 15 and iPhone 15 Plus."
    # ]
    test_queries = [
        "What do users say about iPhone 15 Plus?",
        # "Any complaints about heating issues in iPhone 15?",
        # "How is the display quality of iPhone 15?",
        # "Is iPhone 15 considered value for money?"
    ]
    rag_agent = AgenticRAG()

    for query in test_queries:
        print(f"[query] {query}")
        answer = rag_agent.run(query)
        print("\nFinal Answer:\n", answer)
