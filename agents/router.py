from agents import AgentState

def router_node(state: AgentState):
    query = state.get("query", "").lower()
    
    # We use a lightweight mock router here to prevent Docker from running out of memory (OOM) 
    # when loading large HuggingFace pipelines alongside the other models.
    if "connect" in query or "error code" in query:
        top_intent = "Connectivity/Storage"
    elif "password" in query or "sso" in query:
        top_intent = "Authentication"
    elif "freezing" in query or "slow" in query:
        top_intent = "Performance"
    else:
        top_intent = "General Inquiry"
        
    print(f"[ROUTER] Classified Intent: {top_intent}")
        
    return {"intent": top_intent}
