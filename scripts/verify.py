import os
# Ensure we mock the API key for testing if it's not set
if "OPENAI_API_KEY" not in os.environ:
    os.environ["OPENAI_API_KEY"] = "mock"

from agents.orchestrator import app

def run_test_query(query_text: str):
    print(f"\n{'='*50}\nTesting Query: '{query_text}'\n{'='*50}")
    
    initial_state = {
        "query": query_text,
        "documents": [],
        "draft_answer": "",
        "is_hallucination": False,
        "iterations": 0
    }
    
    # We will invoke the graph and stream the state updates to see the agent flow
    for s in app.stream(initial_state):
        if "retriever" in s:
            print(f"--> [RETRIEVER NODE]")
            docs = s["retriever"]["documents"]
            print(f"    Retrieved {len(docs)} documents.")
        elif "reasoner" in s:
            print(f"--> [REASONER NODE]")
            ans = s["reasoner"]["draft_answer"]
            print(f"    Drafted Answer (preview): {ans[:50]}...")
        elif "verifier" in s:
            print(f"--> [VERIFIER NODE]")
            is_hallucination = s["verifier"]["is_hallucination"]
            print(f"    Hallucination Detected? {is_hallucination}")
            
    print("\n--- Final Output ---")
    final_state = app.invoke(initial_state)
    print(final_state["draft_answer"])

if __name__ == "__main__":
    # Test a normal query
    run_test_query("My CloudStore database won't connect.")
    
    # Test the hallucination retry loop by using the trigger phrase
    run_test_query("Test hallucination: Why is my AuthMagic failing?")
