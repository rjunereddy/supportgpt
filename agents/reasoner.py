from agents import AgentState
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from config.settings import get_settings

def reasoner_node(state: AgentState):
    query = state["query"]
    docs = state.get("documents", [])
    
    settings = get_settings()
    
    # Check keys from state first (user-supplied at runtime), then fallback to settings/.env
    groq_key = state.get("groq_api_key") or settings.groq_api_key or ""
    openai_key = state.get("openai_api_key") or settings.openai_api_key or ""
    
    has_groq = bool(groq_key and groq_key != "mock" and not groq_key.startswith("mock"))
    has_openai = bool(openai_key and openai_key != "mock" and not openai_key.startswith("mock"))
    
    selected_provider = (state.get("provider") or "").lower()
    selected_model = state.get("model")

    # Determine which provider to use
    use_provider = None
    if selected_provider == "groq" and has_groq:
        use_provider = "groq"
    elif selected_provider == "openai" and has_openai:
        use_provider = "openai"
    elif has_groq:
        use_provider = "groq"
    elif has_openai:
        use_provider = "openai"

    # Fallback to mock LLM if no active valid key is provided
    if not use_provider:
        context_preview = docs[0] if docs else "No documents found in knowledge base."
        mock_ans = (
            f"[MOCK LLM RESPONSE — No active API key provided]\n\n"
            f"Based on your query: \"{query}\"\n\n"
            f"Most relevant retrieved context:\n{context_preview}\n\n"
            f"(Provide an OpenAI or Groq API key in the sidebar to generate live LLM responses.)"
        )
        if state.get("iterations", 0) > 0:
            mock_ans = "[MOCK LLM REVISION - Hallucination Loop Tested]\n" + mock_ans
        return {"draft_answer": mock_ans}
        
    try:
        if use_provider == "groq":
            model_name = selected_model or "llama-3.1-8b-instant"
            llm = ChatGroq(model=model_name, temperature=0.1, api_key=groq_key)
        else:
            model_name = selected_model or "gpt-4o-mini"
            llm = ChatOpenAI(model=model_name, temperature=0.1, api_key=openai_key)
        
        # System instructions
        system_msg = (
            "You are SupportGPT, an expert and helpful customer support assistant.\n"
            "Answer the user's inquiry accurately and concisely, relying strictly on the provided Context.\n"
            "If the answer cannot be found in the context, politely state that the current documents do not contain the answer, "
            "and suggest relevant next steps or what documents to upload.\n"
            "Do NOT fabricate information or issue IDs."
        )
        
        if state.get("is_hallucination"):
            system_msg += (
                "\n\nCRITICAL WARNING: The previous response contained ungrounded statements or hallucinations. "
                "Adhere STRICTLY to the provided context and ground every single sentence in the provided facts."
            )

        context_text = "\n\n---\n\n".join(docs) if docs else "No relevant documents found."

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg + "\n\n=== CONTEXT ===\n{context}"),
            ("human", "{question}")
        ])
        
        chain = prompt | llm
        res = chain.invoke({"context": context_text, "question": query})
        return {"draft_answer": res.content}
        
    except Exception as e:
        error_msg = str(e)
        return {
            "draft_answer": (
                f"⚠️ Error calling {use_provider.upper()} API:\n`{error_msg}`\n\n"
                f"Please verify your API key and model selection in the sidebar or `.env`."
            )
        }
