from langgraph.graph import StateGraph, END
from agents import AgentState
from agents.retriever import retrieve_node
from agents.reasoner import reasoner_node
from agents.verifier import verifier_node
from agents.router import router_node

def should_continue(state: AgentState):
    """Router function to determine if we need to regenerate due to hallucinations."""
    if state.get("is_hallucination", False) and state.get("iterations", 0) < 3:
        return "reasoner"
    return END

def build_agent_graph():
    """Builds and compiles the LangGraph state machine."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("reasoner", reasoner_node)
    workflow.add_node("verifier", verifier_node)
    
    workflow.set_entry_point("router")
    workflow.add_edge("router", "retriever")
    workflow.add_edge("retriever", "reasoner")
    workflow.add_edge("reasoner", "verifier")
    
    workflow.add_conditional_edges(
        "verifier", 
        should_continue,
        {
            "reasoner": "reasoner",
            END: END
        }
    )
    
    return workflow.compile()

# Singleton graph application
app = build_agent_graph()
