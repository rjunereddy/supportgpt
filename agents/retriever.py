from agents import AgentState
from retrieval.vector_store import get_retriever
from graph.builder import support_graph

def retrieve_node(state: AgentState):
    query = state["query"]
    retriever = get_retriever()
    
    # 1. Standard Vector Retrieval
    docs = retriever.invoke(query)
    doc_contents = []
    for d in docs:
        source = d.metadata.get("source", "Knowledge Base")
        meta_extra = []
        if "page" in d.metadata:
            meta_extra.append(f"Page {d.metadata['page']}")
        if "row" in d.metadata:
            meta_extra.append(f"Row {d.metadata['row']}")
        if "category" in d.metadata:
            meta_extra.append(f"Category: {d.metadata['category']}")
        
        tag = f"[{source}" + (f" | {', '.join(meta_extra)}" if meta_extra else "") + "]"
        doc_contents.append(f"{tag}\n{d.page_content}")
    
    # 2. Knowledge Graph Augmentation
    products_found = set(d.metadata.get("product") for d in docs if "product" in d.metadata and d.metadata.get("product"))
    
    kg_context = []
    for product in products_found:
        related_info = support_graph.get_related_issues(product)
        if "No information found" not in related_info:
            kg_context.append(f"[KNOWLEDGE GRAPH: {product}]\n{related_info}")
        
    # Combine FAISS docs with Graph docs
    all_context = doc_contents + kg_context
        
    return {"documents": all_context}
