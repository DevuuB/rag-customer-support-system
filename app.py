import traceback
from rag_pipeline import create_vectorstore, get_retriever
from graph_flow import build_graph
from hitl import human_intervention

print("🔄 Loading system...")

# Create DB (only once)
create_vectorstore("data/sample.pdf")

retriever = get_retriever()
graph = build_graph()

print("\n✅ Customer Support Assistant Ready!\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    try:
        state = {
            "query": query,
            "context": "",
            "answer": "",
            "confidence": 0.0,
            "status": "",
            "retriever": retriever
        }

        # 👇 MANUAL FLOW (instead of graph.invoke)
        from graph_flow import processing_node, decision_node

        state = processing_node(state)
        state = decision_node(state)

        if state["status"] == "HITL":
            answer = human_intervention(query)
            print("Bot:", answer)
        else:
            print("Bot:", state["answer"])


    except Exception as e:
        print("❌ FULL ERROR:")
        traceback.print_exc()



    