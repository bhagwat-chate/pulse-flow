# prod_assistant/workflow/agentic_rag_workflow.py

from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import START, END
from langgraph.graph.message import add_messages, BaseMessage
from prod_assistant.prompt_library.prompts import PROMPT_REGISTRY

from prod_assistant.retriever.retrieval import Retriever
from prod_assistant.utils.model_loader import ModelLoader
from langgraph.checkpoint.memory import MemorySaver


class AgenticRAG:

    class AgenticState(TypedDict):
        messages: Annotated[Sequence[BaseMessage], add_messages]

    def __init__(self):
        self.retriever_obj = Retriever()
        self.model_loader = ModelLoader()
        self.llm = self.model_loader.load_llm()
        self.checkpointer = MemorySaver()
        self.workflow = self._build_workflow()
        self.app = self.workflow.compile(checkpointer=self.checkpointer)

    def _format_docs(self, docs) -> str:
        if not docs:
            return "no relevant documents found"
        formatted_chunks = []

        for d in docs:
            meta = docs.metadata or {}

            formatted = (
                f"Title: {meta.get('product_title'), 'N/A'}"
                f"Price: {meta.get('price'), 'N/A'}"
                f"Rating: {meta.get('rating'), 'N/A'}"
                f"Reviews: \n{meta.get(d.page_content.strip())}"
            )
        formatted_chunks.append(formatted)

        return "\n\n---\n\n".join(formatted_chunks)

    def _ai_assistant(self, state: AgenticState):
        print("---CALL ASSISTANT---")

        message = state['messages']
        last_message = message[-1].content

        if any(word in last_message.lower() for word in ['price', 'review', 'product']):
            return {"messages": [HumanMessage(content="TOOL: retriever")]}
        else:
            prompt = ChatPromptTemplate.from_template(
                "You are an helpful assistant. Help the user directly.\n\nQuestion: {question}\nAnswer:"
            )
            chain = prompt | self.llm | StrOutputParser()

            response = chain.invoke({"question": last_message})

            return {"messages": [HumanMessage(content=response)]}

