import asyncio
from typing import Sequence, Annotated, TypedDict, Literal
from langchain_core.messages import HumanMessage, BaseMessage
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY, PromptType
from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver
from prod_assistant.evaluation.ragas_eval import evaluate_context_precision, evaluate_response_relevancy
from langchain_mcp_adapters.client import MultiServerMCPClient


class AgenticRAG:

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader_obj = ModelLoader()
        self.llm = self.model_loader_obj.load_llm()
        self.checkpointer = MemorySaver()

        self.mcp_client = MultiServerMCPClient({
            "hybrid_search": {
                "command": "python",
                "args": ["-m", "prod_assistant.mcp_servers.product_search_server"],
                "transport": "stdio"
            }
        })
        self.mcp_tools = asyncio.run(self.mcp_client.get_tools())
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _ai_assistant(self, state: AgentState):
        print("---ASSISTANT---")
        messages = state['messages'][-1].content

        if any(word in messages.lower() for word in ['price', 'review', 'product']):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_messages(
                "You are a helpful assistant. Answer the user directly. \nQuestion:{question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()
            response = chain.invoke({"question": messages})

            return {"messages": [HumanMessage(content=response)]}

    def _vector_retriever(self, state: AgentState):
        print("---RETRIEVER (MCP)---")
        query = state['messages'][0].content
        tool = next(t for t in self.mcp_tools if t.name == 'get_product_info')
        result = asyncio.run(tool.ainvoke({"query": query}))
        context = result if result else "No data"

        return {"messages": [HumanMessage(content=context)]}

    def _web_search(self, state: AgentState):
        print("---WEB SEARCH---")
        query = state['messages'][0].content
        tool = any(t for t in self.mcp_tools if t.name == 'web_search')
        result = asyncio.run(tool.ainvoke({"query": query}))
        context = result if result else "No data from web"

        return {"messages": [HumanMessage(content=context)]}

    def _grade_documents(self, state: AgentState) -> Literal["generator", "rewriter"]:
        print("---GRADER---")
        question = state['messages'][0].content
        docs = state['messages'][-1].content

        prompt = PromptTemplate(
            template="""You area a grader. Question: {question}\nDocs: {docs}\nAre docs relevant to the question? 
            Answer yes or no""",
            input_variables=['question', 'docs']
        )

        chain = prompt | self.llm | StrOutputParser()

        score = chain.invoke({"question": question, "docs": docs})

        return "generator" if "yes" in score.lower() else "rewriter"

    def _generate(self, state: AgentState):
        print("---GENERATE---")
        question = state['messages'][0].content
        docs = state['messages'][-1].content

        prompt = ChatPromptTemplate.from_template(
            PROMPT_REGISTRY[PromptType.PRODUCT_BOT].template
        )
        chain = prompt | self.llm | StrOutputParser()

        response = chain.invoke({'question': question, "docs": docs})

        return {"messages": [HumanMessage(content=response)]}

