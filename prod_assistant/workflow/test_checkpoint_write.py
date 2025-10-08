"""
Async Diagnostic — LangGraph PostgresSaver (v0.6.9+)
Author: Bhagwat Chate
Purpose: Verify async checkpoint persistence in AWS RDS
"""

import asyncio
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.postgres import PostgresSaver


class AgentState(dict):
    pass


def simple_node(state: AgentState):
    print(f"⚙️ Running async node: {state}")
    return {"messages": [HumanMessage(content="Hello async world!")]}


async def main():
    db_uri = (
        "postgresql://pulseflowdb:$DBPulaws2025"
        "@database-1.cxwsq62600ku.ap-south-1.rds.amazonaws.com:5432/postgres"
    )

    # ✅ PostgresSaver is now async
    async with PostgresSaver.from_conn_string(db_uri) as saver:
        await saver.setup()
        print("✅ Connected to AWS RDS and ensured schema migration")

        workflow = StateGraph(AgentState)
        workflow.add_node("SimpleNode", simple_node)
        workflow.add_edge(START, "SimpleNode")
        workflow.add_edge("SimpleNode", END)

        app = workflow.compile(checkpointer=saver)

        print("🚀 Running async LangGraph workflow with checkpoint persistence...")
        result = await app.ainvoke(
            {"messages": [HumanMessage(content="Test async checkpoint save")]},
            config={
                "configurable": {
                    "thread_id": "async_checkpoint_test",
                    "checkpoint_ns": "diagnostics",
                }
            },
        )

        print("✅ Run complete:", result)


if __name__ == "__main__":
    asyncio.run(main())
