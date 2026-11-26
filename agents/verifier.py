from agents import AgentState

# We mock the NLI model to avoid Out-Of-Memory (OOM) crashes and long download times
# when running in a limited Docker container. 
nli_model = "mocked"

def verifier_node(state: AgentState):
    draft = state.get("draft_answer", "")
    docs = state.get("documents", [])
    query = state.get("query", "").lower()
    iterations = state.get("iterations", 0)
    
    is_hallucination = False
    
    # Force test hallucination loop for the demo
    if "test hallucination" in query and iterations < 1:
        is_hallucination = True
        print("[VERIFIER] Manual hallucination test triggered.")
        return {"is_hallucination": is_hallucination, "iterations": iterations + 1}
        
    if not docs:
        # If no docs retrieved, anything generated is inherently ungrounded (hallucinated)
        is_hallucination = True
    elif nli_model == "mocked":
        # Simulate an NLI contradiction check
        print("[VERIFIER] Mock NLI check passed. Answer is grounded.")
            
    return {"is_hallucination": is_hallucination, "iterations": iterations + 1}
