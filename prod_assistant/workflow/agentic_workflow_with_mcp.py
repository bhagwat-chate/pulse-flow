# import asyncio
# from typing import Annotated, Sequence, TypedDict, Literal
#
# from langchain_core.messages import HumanMessage, BaseMessage
# from langchain.prompts import PromptTemplate, ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langgraph.graph import START, END, StateGraph
# from langgraph.graph.message import add_messages
#
# from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
# from prod_assistant.retriever.retrieval import Retriever
# from prod_assistant.utils.model_loader import ModelLoader
# from langgraph.checkpoint.memory import MemorySaver
# from langchain_mcp_adapters.client import MultiServerMCPClient
#
#
# class AgenticRAG:
#
#     class AgentState(TypedDict):
#         messages: Annotated[Sequence[BaseMessage], add_messages]
#
#     def __init__(self):
#         # Core components
#         self.retriever_obj = Retriever()
#         self.model_loader_obj = ModelLoader()
#         self.llm = self.model_loader_obj.load_llm()
#         self.checkpointer = MemorySaver()
#
#         # MCP client setup
#         self.mcp_client = MultiServerMCPClient({
#             "product_retriever": {
#                 "command": "python",
#                 "args": ["-m", "prod_assistant.mcp_servers.product_search_server"],
#                 "transport": "stdio"
#             }
#         })
#
#         # Load all available MCP tools
#         self.mcp_tools = asyncio.run(self.mcp_client.get_tools())
#
#         # Compile workflow
#         self.workflow = self._build_workflow()
#         self.app = self.workflow.compile(checkpointer=self.checkpointer)
#
#     # ------------------------- NODES --------------------
#
#     def _ai_assistants(self, state: AgentState):
#         """Decides whether to call retriever tool or answer directly."""
#         print("---CALL ASSISTANT---")
#         messages = state['messages']
#         last_message = messages[-1].content
#
#         if any(word in last_message.lower() for word in ['price', 'review', 'product']):
#             return {"messages": [HumanMessage(content="TOOL: retriever")]}
#         else:
#             prompt = ChatPromptTemplate.from_template(
#                 "You are a helpful assistant. Answer the user directly.\n\n"
#                 "Question: {question}\nAnswer: "
#             )
#             chain = prompt | self.llm | StrOutputParser()
#             response = chain.invoke({"question": last_message})
#             return {"messages": [HumanMessage(content=response)]}
#
#     def _vector_retriever(self, state: AgentState):
#         """Call MCP retriever. Fallback to web_search if no local results."""
#         print("---RETRIEVER (MCP)---")
#         query = state['messages'][-1].content
#
#         # Get tools
#         get_product_info = next(t for t in self.mcp_tools if t.name == 'get_product_info')
#         web_search = next(t for t in self.mcp_tools if t.name == 'web_search')
#
#         # Call retriever
#         result = asyncio.run(get_product_info.ainvoke({"query": query}))
#         if "No local results found." in result:
#             print("⚠️ No local results, falling back to web_search...")
#             result = asyncio.run(web_search.ainvoke({"query": query}))
#
#         return {"messages": [HumanMessage(content=result)]}
#
#     def _grade_documents(self, state: AgentState) -> Literal["Generator", "Rewriter"]:
#         """Grade docs: relevant → Generator, else → Rewriter."""
#         print("--- GRADER ---")
#         question = state["messages"][0].content
#         docs = state["messages"][-1].content
#
#         prompt = PromptTemplate(
#             template="""You are a grader. Question: {question}\nDocs: {docs}\n
#             Are docs relevant to the question? Answer yes or no.""",
#             input_variables=["question", "docs"],
#         )
#         chain = prompt | self.llm | StrOutputParser()
#         score = chain.invoke({"question": question, "docs": docs})
#         return "Generator" if "yes" in score.lower() else "Rewriter"
#
#     def _generate(self, state: AgentState):
#         """Generate final answer using context + question."""
#         print("---GENERATE---")
#         question = state['messages'][0].content
#         docs = state['messages'][-1].content
#
#         prompt = ChatPromptTemplate.from_template(
#             PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
#         )
#         chain = prompt | self.llm | StrOutputParser()
#         response = chain.invoke({"context": docs, "question": question})
#
#         return {"messages": HumanMessage(content=response)}
#
#     def _rewrite(self, state: AgentState):
#         """Rewrite query if docs are irrelevant."""
#         print("---REWRITE QUERY---")
#         question = state['messages'][0].content
#         prompt = ChatPromptTemplate.from_template(
#             "Rewrite this user query to make it more clear and specific for a search engine.\n"
#             "Do not answer the query. Only rewrite it.\n\nQuery:{question}\nRewritten Query:"
#         )
#         chain = prompt | self.llm | StrOutputParser()
#         response = chain.invoke({'question': question})
#         return {"messages": HumanMessage(content=response)}
#
#     # ------------------------- WORKFLOW --------------------
#
#     def _build_workflow(self):
#         workflow = StateGraph(self.AgentState)
#
#         workflow.add_node("Assistant", self._ai_assistants)
#         workflow.add_node("Retriever", self._vector_retriever)
#         workflow.add_node("Generator", self._generate)
#         workflow.add_node("Rewriter", self._rewrite)
#
#         # Flow: START → Assistant
#         workflow.add_edge(START, "Assistant")
#
#         # Assistant decides → Retriever or END
#         workflow.add_conditional_edges(
#             "Assistant",
#             lambda state: "Retriever" if "TOOL" in state['messages'][-1].content else END,
#             {"Retriever": "Retriever", END: END}
#         )
#
#         # Retriever → Generator / Rewriter
#         workflow.add_conditional_edges(
#             "Retriever",
#             self._grade_documents,
#             {"Generator": "Generator", "Rewriter": "Rewriter"}
#         )
#
#         workflow.add_edge("Generator", END)
#         workflow.add_edge("Rewriter", "Assistant")
#
#         return workflow
#
#     # ------------------------- RUN --------------------
#
#     def run(self, query: str, thread_id: str = 'default_thread') -> str:
#         """Run the workflow with a query."""
#         result = self.app.invoke(
#             {"messages": [HumanMessage(content=query)]},
#             config={"configurable": {'thread_id': thread_id}}
#         )
#         return result['messages'][-1].content
#
#
# if __name__ == '__main__':
#     rag_agent = AgenticRAG()
#     answer = rag_agent.run("What is the price of iPhone 15 plus?")
#     print(f"\nFinal Answer: {answer}")


import asyncio
from typing import Annotated, Sequence, TypedDict, Literal

from langchain_core.messages import HumanMessage, BaseMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages

from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
from langchain_mcp_adapters.client import MultiServerMCPClient


class AgenticRAG:
    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]
        did_web_search: bool
        did_rewrite: bool

    def __init__(self):
        # Core components
        self.retriever_obj = Retriever()
        self.model_loader_obj = ModelLoader()
        self.llm = self.model_loader_obj.load_llm()
        self.checkpointer = MemorySaver()

        # MCP client setup
        self.mcp_client = MultiServerMCPClient({
            "product_retriever": {
                "command": "python",
                "args": ["-m", "prod_assistant.mcp_servers.product_search_server"],
                "transport": "stdio"
            }
        })

        # Load all available MCP tools
        self.mcp_tools = asyncio.run(self.mcp_client.get_tools())

        # Compile workflow
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    # ------------------------- NODES --------------------

    def _ai_assistants(self, state: AgentState):
        """Decides whether to call retriever tool or answer directly."""
        print("---CALL ASSISTANT---")
        messages = state['messages']
        last_message = messages[-1].content
        print(f"[ASSISTANT] Processing message: {last_message}")

        if any(word in last_message.lower() for word in ['price', 'review', 'product']):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are a helpful assistant. Answer the user directly.\n\n"
                "Question: {question}\nAnswer: "
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": last_message})
            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("---RETRIEVER (MCP)---")
        query = state['messages'][-1].content
        print(f"[RETRIEVER (MCP)] query: {query}")

        get_product_info = next(t for t in self.mcp_tools if t.name == 'get_product_info')
        web_search = next(t for t in self.mcp_tools if t.name == 'web_search')

        # Try local retriever
        result = asyncio.run(get_product_info.ainvoke({"query": query}))

        if "No local results found." in result and not state.get("did_web_search", False):
            print("⚠️ No local results, falling back to web_search (only once)...")
            result = asyncio.run(web_search.ainvoke({"query": query}))
            return {"messages": [HumanMessage(content=result)], "did_web_search": True}

        return {"messages": [HumanMessage(content=result)]}

    def _grade_documents(self, state: AgentState) -> Literal["Generator", "Rewriter"]:
        """Grade docs: relevant → Generator, else → Rewriter."""
        print("--- GRADER ---")
        question = state["messages"][0].content
        docs = state["messages"][-1].content
        print(f"[GRADER] question: {question}")
        print(f"[GRADER] docs: {docs}")

        prompt = PromptTemplate(
            template="""You are a grader. Question: {question}\nDocs: {docs}\n
            Are docs relevant to the question? Answer yes or no.""",
            input_variables=["question", "docs"],
        )
        chain = prompt | self.llm | StrOutputParser()
        score = chain.invoke({"question": question, "docs": docs})
        return "Generator" if "yes" in score.lower() else "Rewriter"

    def _generate(self, state: AgentState):
        """Generate final answer using context + question."""
        print("---GENERATE---")
        question = state['messages'][0].content
        docs = state['messages'][-1].content
        print(f"[GENERATE] docs: {question}")
        print(f"[GENERATE] docs: {docs}")

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({"context": docs, "question": question})

        return {"messages": HumanMessage(content=response)}

    def _rewrite(self, state: AgentState):
        print("---REWRITE QUERY---")

        if state.get("did_rewrite", False):
            # Already rewritten once, stop loop
            return {"messages": [HumanMessage(content="No relevant results found. Please try another query.")]}

        question = state['messages'][0].content
        print(f"[REWRITE QUERY] question: {question}")

        prompt = ChatPromptTemplate.from_template(
            "Rewrite this user query to make it more clear and specific for a search engine.\n"
            "Do not answer the query. Only rewrite it.\n\nQuery:{question}\nRewritten Query:"
        )
        print(f"[REWRITE QUERY] prompt: {prompt}")

        chain = prompt | self.llm | StrOutputParser()
        response = chain.invoke({'question': question})
        print(f"[REWRITE QUERY] response: {response}")

        return {
            "messages": [HumanMessage(content=response)],
            "did_rewrite": True
        }

    # ------------------------- WORKFLOW --------------------

    def _build_workflow(self):
        workflow = StateGraph(self.AgentState)

        workflow.add_node("Assistant", self._ai_assistants)
        workflow.add_node("Retriever", self._vector_retriever)
        workflow.add_node("Generator", self._generate)
        workflow.add_node("Rewriter", self._rewrite)

        # Flow: START → Assistant
        workflow.add_edge(START, "Assistant")

        # Assistant decides → Retriever or END
        workflow.add_conditional_edges(
            "Assistant",
            lambda state: "Retriever" if "TOOL" in state['messages'][-1].content else END,
            {"Retriever": "Retriever", END: END}
        )

        # Retriever → Generator / Rewriter
        workflow.add_conditional_edges(
            "Retriever",
            self._grade_documents,
            {"Generator": "Generator", "Rewriter": "Rewriter"}
        )

        workflow.add_edge("Generator", END)
        workflow.add_edge("Rewriter", "Assistant")

        return workflow

    # ------------------------- RUN --------------------

    def run(self, query: str, thread_id: str = 'default_thread') -> str:
        """Run the workflow with a query."""
        result = self.app.invoke(
            {"messages": [HumanMessage(content=query)],
             "did_web_search": False,
             "did_rewrite": False},
            config={"configurable": {'thread_id': thread_id}}
        )

        return result['messages'][-1].content


if __name__ == '__main__':
    rag_agent = AgenticRAG()
    query = "iPhone 15 Plus price?"
    print(f"[query]: {query}")
    answer = rag_agent.run(query)
    print(f"\nFinal Answer: {answer}")
