import streamlit as st
import requests
import os
import json

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="SupportGPT — Multi-Agent RAG", page_icon="🤖", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "groq_api_key" not in st.session_state:
    st.session_state.groq_api_key = ""
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""
if "provider" not in st.session_state:
    st.session_state.provider = "Groq"
if "model" not in st.session_state:
    st.session_state.model = "llama-3.1-8b-instant"

# Helper function to check API status and stats
def fetch_api_health_and_stats():
    try:
        health_res = requests.get(f"{API_URL}/", timeout=3)
        health = health_res.json() if health_res.status_code == 200 else {}
    except Exception:
        health = {}

    try:
        stats_res = requests.get(f"{API_URL}/stats", timeout=3)
        stats = stats_res.json() if stats_res.status_code == 200 else {}
    except Exception:
        stats = {}
    return health, stats

health_data, stats_data = fetch_api_health_and_stats()

# ================= SIDEBAR =================
with st.sidebar:
    st.title("⚙️ SupportGPT Controls")
    
    # 1. API Keys & LLM Provider Configuration
    st.subheader("🔑 LLM & API Configuration")
    
    provider_choice = st.radio(
        "Select LLM Provider",
        options=["Groq", "OpenAI"],
        index=0 if st.session_state.provider == "Groq" else 1,
        help="Groq offers ultra-fast responses with free tier options. OpenAI requires an OpenAI API key."
    )
    st.session_state.provider = provider_choice

    if provider_choice == "Groq":
        model_options = [
            "llama-3.1-8b-instant",
            "llama-3.1-70b-versatile",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        # Preserve user selection if valid, otherwise default to first
        current_idx = model_options.index(st.session_state.model) if st.session_state.model in model_options else 0
        selected_model = st.selectbox("Select Groq-Hosted Model", model_options, index=current_idx)
        st.session_state.model = selected_model

        # Show key input
        groq_input = st.text_input(
            "Groq API Key",
            value=st.session_state.groq_api_key,
            type="password",
            placeholder="gsk_...",
            help="Get your key from https://console.groq.com/keys"
        )
        if groq_input:
            st.session_state.groq_api_key = groq_input

        persist_key = st.checkbox("Save key to .env for future sessions", value=True)
        if st.button("💾 Apply Groq Key", use_container_width=True):
            if st.session_state.groq_api_key:
                try:
                    res = requests.post(
                        f"{API_URL}/config/keys",
                        json={"groq_api_key": st.session_state.groq_api_key, "persist": persist_key},
                        timeout=5
                    )
                    if res.status_code == 200:
                        st.success("Groq API Key applied!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to communicate with API: {e}")
            else:
                st.warning("Please enter a valid key first.")

    else:
        model_options = [
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-3.5-turbo"
        ]
        current_idx = model_options.index(st.session_state.model) if st.session_state.model in model_options else 0
        selected_model = st.selectbox("Select OpenAI Model", model_options, index=current_idx)
        st.session_state.model = selected_model

        openai_input = st.text_input(
            "OpenAI API Key",
            value=st.session_state.openai_api_key,
            type="password",
            placeholder="sk-...",
            help="Get your key from https://platform.openai.com/api-keys"
        )
        if openai_input:
            st.session_state.openai_api_key = openai_input

        persist_key = st.checkbox("Save key to .env for future sessions", value=True)
        if st.button("💾 Apply OpenAI Key", use_container_width=True):
            if st.session_state.openai_api_key:
                try:
                    res = requests.post(
                        f"{API_URL}/config/keys",
                        json={"openai_api_key": st.session_state.openai_api_key, "persist": persist_key},
                        timeout=5
                    )
                    if res.status_code == 200:
                        st.success("OpenAI API Key applied!")
                        st.rerun()
                except Exception as e:
                    st.error(f"Failed to communicate with API: {e}")
            else:
                st.warning("Please enter a valid key first.")

    # Status indication
    is_groq_active = health_data.get("groq_configured", False) or bool(st.session_state.groq_api_key)
    is_openai_active = health_data.get("openai_configured", False) or bool(st.session_state.openai_api_key)
    
    st.markdown("---")
    st.markdown("**Backend Status:**")
    if provider_choice == "Groq" and is_groq_active:
        st.caption("🟢 Active: **Groq** (" + st.session_state.model + ")")
    elif provider_choice == "OpenAI" and is_openai_active:
        st.caption("🟢 Active: **OpenAI** (" + st.session_state.model + ")")
    else:
        st.caption("🟡 **Mock LLM Mode** (Enter API key above for live models)")

    st.markdown("---")

    # 2. Document Upload & Knowledge Base Management
    st.subheader("📚 Upload Support Documents")
    st.markdown("Upload documents to enrich the Knowledge Base in real time.")

    uploaded_files = st.file_uploader(
        "Supported: .txt, .md, .pdf, .csv, .json",
        type=["txt", "md", "pdf", "csv", "json"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if st.button("🚀 Ingest & Index Documents", use_container_width=True, type="primary"):
            with st.spinner("Processing, chunking, and embedding documents..."):
                try:
                    files_payload = [
                        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
                        for f in uploaded_files
                    ]
                    upload_res = requests.post(f"{API_URL}/upload", files=files_payload, timeout=60)
                    if upload_res.status_code == 200:
                        res_json = upload_res.json()
                        st.success(f"Indexed {res_json.get('chunks_added', 0)} chunks successfully!")
                        st.rerun()
                    else:
                        st.error(f"Upload failed ({upload_res.status_code}): {upload_res.text}")
                except Exception as e:
                    st.error(f"Upload error: {str(e)}")

    # Index details
    total_chunks = stats_data.get("total_chunks", 0)
    files_list = stats_data.get("uploaded_files", [])
    st.metric(label="Total Chunks in Knowledge Base", value=total_chunks)

    if files_list:
        with st.expander(f"📁 Indexed Files ({len(files_list)})", expanded=False):
            for file_info in files_list:
                fname = file_info.get("filename", "Unknown")
                fchunks = file_info.get("chunks", 0)
                ftime = file_info.get("uploaded_at", "")
                st.write(f"- **{fname}** ({fchunks} chunks) \n  *{ftime}*")

    if st.button("🔄 Reset to Default Baseline Index", use_container_width=True):
        try:
            reset_res = requests.post(f"{API_URL}/reset-index", timeout=10)
            if reset_res.status_code == 200:
                st.info("Knowledge Base reset to default.")
                st.rerun()
        except Exception as e:
            st.error(f"Reset failed: {e}")

# ================= MAIN CHAT AREA =================
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🤖 SupportGPT Agent")
    st.markdown("Agentic RAG Assistant powered by **LangGraph**, **FAISS Vector Retrieval**, and **Self-RAG Hallucination Verification**.")
with col2:
    current_prov = st.session_state.provider
    current_mod = st.session_state.model
    st.info(f"**Provider**: {current_prov}\n\n**Model**: `{current_mod}`")

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User prompt input
if prompt := st.chat_input("Ask a support question (e.g. 'How do I resolve database connection timeouts?'):"):
    # Append user question
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke agent backend
    with st.chat_message("assistant"):
        with st.spinner("Retrieving knowledge, reasoning, and fact-checking..."):
            try:
                # Prepare query payload with active keys and model
                payload = {
                    "query": prompt,
                    "provider": st.session_state.provider.lower(),
                    "model": st.session_state.model,
                    "groq_api_key": st.session_state.groq_api_key or None,
                    "openai_api_key": st.session_state.openai_api_key or None
                }

                response = requests.post(f"{API_URL}/query", json=payload, timeout=60)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display Agent Thought Process & Grounding Evidence
                    with st.expander("🔍 Agent Thought Process & Grounding Evidence", expanded=False):
                        st.write(f"📚 **Documents Retrieved**: {data.get('documents_retrieved', 0)}")
                        st.write(f"🔄 **Self-Reflective Iterations**: {data.get('iterations', 0)}")
                        
                        if data.get('hallucination_detected'):
                            st.warning("⚠️ Hallucination detected during initial draft! The verifier prompted a self-correction.")
                        else:
                            st.success("✅ Verifier confirmed answer is factually grounded in context.")
                        
                        sources = data.get("sources", [])
                        if sources:
                            st.markdown("#### Retrieved Context Sources:")
                            for i, src in enumerate(sources):
                                st.markdown(f"**Source {i+1}:**")
                                st.code(src, language="text")
                        
                    # Display Answer
                    answer = data.get("answer", "No answer generated.")
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                else:
                    st.error(f"API Error ({response.status_code}): {response.text}")
            except requests.exceptions.ConnectionError:
                st.error(f"Could not connect to FastAPI server at `{API_URL}`. Make sure `uvicorn api.main:app` is running!")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
