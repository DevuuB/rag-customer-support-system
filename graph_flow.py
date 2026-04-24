from typing import TypedDict
from langgraph.graph import StateGraph
from groq import Groq
from dotenv import load_dotenv
import os

from config import CONFIDENCE_THRESHOLD

# Load .env file
load_dotenv()

# Debug
print("GROQ KEY:", os.getenv("GROQ_API_KEY"))

# ✅ Groq Client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

# ✅ State structure
class State(TypedDict):
    query: str
    context: str
    answer: str
    confidence: float
    status: str
    retriever: object


# 🔹 Processing Node
def processing_node(state):
    retriever = state["retriever"]

    docs = retriever.invoke(state["query"])

    # Safe context
    if not docs:
        context = ""
    else:
        context = "\n".join([d.page_content for d in docs])

    prompt = f"""
Use the context below to answer the question.
If answer not found, say 'I don't know'.

Context:
{context}

Question:
{state['query']}
"""

    # ✅ GROQ CHAT CALL (FIXED)
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )

    answer = response.choices[0].message.content

    # Store answer
    state["answer"] = answer

    # ✅ Confidence logic
    if not context.strip() or "I don't know" in answer:
        state["confidence"] = 0.3
    else:
        state["confidence"] = 0.85

    return state


# 🔹 Decision Node
def decision_node(state):
    if state["confidence"] < CONFIDENCE_THRESHOLD:
        state["status"] = "HITL"
    else:
        state["status"] = "SUCCESS"
    return state


# 🔹 Build Graph
def build_graph():
    graph = StateGraph(State)

    graph.add_node("process", processing_node)
    graph.add_node("decision", decision_node)

    graph.set_entry_point("process")
    graph.add_edge("process", "decision")
    graph.set_finish_point("decision")

    return graph.compile()