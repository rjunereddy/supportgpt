import os
import io
import csv
import json
import time
from typing import List, Dict, Any, Optional
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        # Fallback simple text splitter if langchain splitter is missing
        class RecursiveCharacterTextSplitter:
            def __init__(self, chunk_size: int = 600, chunk_overlap: int = 80):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap

            def split_text(self, text: str) -> List[str]:
                if not text:
                    return []
                chunks = []
                step = max(1, self.chunk_size - self.chunk_overlap)
                for i in range(0, len(text), step):
                    chunk = text[i:i + self.chunk_size].strip()
                    if chunk:
                        chunks.append(chunk)
                return chunks

            def split_documents(self, documents: List[Document]) -> List[Document]:
                result = []
                for doc in documents:
                    text_chunks = self.split_text(doc.page_content)
                    for i, tc in enumerate(text_chunks):
                        meta = dict(doc.metadata)
                        meta["chunk"] = i
                        result.append(Document(page_content=tc, metadata=meta))
                return result

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'raw', 'support_tickets.json')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
FAISS_INDEX_PATH = os.path.join(PROCESSED_DIR, 'faiss_index')
MANIFEST_PATH = os.path.join(PROCESSED_DIR, 'manifest.json')

_embeddings_instance = None

def get_embeddings():
    """Lazily load and cache HuggingFaceEmbeddings to avoid repeated heavy model loads."""
    global _embeddings_instance
    if _embeddings_instance is None:
        _embeddings_instance = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return _embeddings_instance

def load_manifest() -> Dict[str, Any]:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {"total_chunks": 0, "uploaded_files": []}

def save_manifest(manifest: Dict[str, Any]):
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2)

def build_baseline_documents() -> List[Document]:
    """Builds initial documents from raw support tickets if available."""
    documents = []
    if os.path.exists(RAW_PATH):
        try:
            with open(RAW_PATH, 'r', encoding='utf-8') as f:
                tickets = json.load(f)
            for t in tickets:
                content = f"Product: {t.get('product', 'General')}\nCategory: {t.get('category', 'General')}\nIssue: {t.get('issue', '')}\nResolution: {t.get('resolution', '')}"
                metadata = {
                    "id": t.get("id", "ticket"),
                    "source": "support_tickets.json",
                    "product": t.get("product", "General"),
                    "category": t.get("category", "General")
                }
                documents.append(Document(page_content=content, metadata=metadata))
        except Exception as e:
            print(f"Error reading baseline support tickets: {e}")

    if not documents:
        # Default placeholder document so vector store is never empty
        documents.append(Document(
            page_content="SupportGPT Knowledge Base initialized. Upload documents to enrich search context.",
            metadata={"source": "system_init", "category": "General"}
        ))
    return documents

def ingest_data() -> FAISS:
    """Builds the initial FAISS index from baseline documents and saves it."""
    documents = build_baseline_documents()
    embeddings = get_embeddings()
    print(f"Building initial FAISS index with {len(documents)} documents...")
    vector_store = FAISS.from_documents(documents, embeddings)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)

    manifest = {
        "total_chunks": len(documents),
        "uploaded_files": [{"filename": "support_tickets.json", "chunks": len(documents), "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")}]
    }
    save_manifest(manifest)
    print(f"Saved initial FAISS index to {FAISS_INDEX_PATH}")
    return vector_store

def load_vector_store() -> FAISS:
    """Loads existing FAISS index or creates a fresh one."""
    embeddings = get_embeddings()
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            return FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            print(f"Error loading FAISS index: {e}. Re-ingesting...")
    return ingest_data()

def get_retriever(k: int = 3):
    vector_store = load_vector_store()
    return vector_store.as_retriever(search_kwargs={"k": k})

def parse_file_to_documents(filename: str, file_bytes: bytes) -> List[Document]:
    """Parses text, markdown, pdf, csv, or json bytes into LangChain Documents with chunking."""
    ext = os.path.splitext(filename)[1].lower()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=80)
    raw_docs: List[Document] = []

    if ext in [".txt", ".md", ".log"]:
        try:
            text = file_bytes.decode('utf-8')
        except UnicodeDecodeError:
            text = file_bytes.decode('latin-1', errors='replace')
        raw_docs.append(Document(page_content=text, metadata={"source": filename, "type": ext}))

    elif ext == ".pdf":
        text = ""
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    raw_docs.append(Document(
                        page_content=page_text,
                        metadata={"source": filename, "page": page_num + 1, "type": "pdf"}
                    ))
        except Exception as e:
            # Fallback if pypdf has an issue or is not installed
            print(f"Warning: pypdf extraction failed ({e}), extracting readable ASCII.")
            try:
                readable = "".join([chr(b) for b in file_bytes if 32 <= b <= 126 or b in (10, 13)])
                raw_docs.append(Document(page_content=readable, metadata={"source": filename, "type": "pdf_fallback"}))
            except Exception:
                pass

    elif ext == ".csv":
        try:
            content_str = file_bytes.decode('utf-8', errors='replace')
            reader = csv.DictReader(io.StringIO(content_str))
            for idx, row in enumerate(reader):
                if "issue" in row and "resolution" in row:
                    content = f"Product: {row.get('product', 'General')}\nIssue: {row.get('issue', '')}\nResolution: {row.get('resolution', '')}"
                else:
                    content = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                if content.strip():
                    raw_docs.append(Document(
                        page_content=content,
                        metadata={"source": filename, "row": idx, "type": "csv"}
                    ))
        except Exception as e:
            print(f"CSV parsing error: {e}")

    elif ext == ".json":
        try:
            content_str = file_bytes.decode('utf-8', errors='replace')
            data = json.loads(content_str)
            if isinstance(data, list):
                for idx, item in enumerate(data):
                    if isinstance(item, dict):
                        if "issue" in item and "resolution" in item:
                            content = f"Product: {item.get('product', 'General')}\nIssue: {item.get('issue', '')}\nResolution: {item.get('resolution', '')}"
                        else:
                            content = json.dumps(item, indent=2)
                    else:
                        content = str(item)
                    raw_docs.append(Document(page_content=content, metadata={"source": filename, "item": idx, "type": "json"}))
            elif isinstance(data, dict):
                raw_docs.append(Document(page_content=json.dumps(data, indent=2), metadata={"source": filename, "type": "json"}))
        except Exception as e:
            print(f"JSON parsing error: {e}")

    else:
        # Generic text attempt for unrecognized extensions
        try:
            text = file_bytes.decode('utf-8', errors='replace')
            raw_docs.append(Document(page_content=text, metadata={"source": filename, "type": "raw"}))
        except Exception:
            pass

    # Split documents into optimal chunks for embedding
    split_docs = text_splitter.split_documents(raw_docs)
    return split_docs

def add_documents_to_index(new_docs: List[Document]) -> int:
    """Adds documents to the FAISS index and saves to disk."""
    if not new_docs:
        return 0

    vector_store = load_vector_store()
    vector_store.add_documents(new_docs)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    vector_store.save_local(FAISS_INDEX_PATH)
    return len(new_docs)

def ingest_uploaded_files(files_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Saves uploaded files to data/uploads/, parses them, chunks them,
    and indexes them in FAISS.
    Each item in files_data should have:
      - 'filename': str
      - 'content': bytes
    """
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    manifest = load_manifest()
    
    total_added_chunks = 0
    file_summaries = []

    for item in files_data:
        filename = item["filename"]
        content = item["content"]

        # 1. Save copy to data/uploads/ for persistence
        saved_path = os.path.join(UPLOADS_DIR, filename)
        try:
            with open(saved_path, 'wb') as f:
                f.write(content)
        except Exception as e:
            print(f"Could not save file to uploads folder: {e}")

        # 2. Parse and chunk
        chunks = parse_file_to_documents(filename, content)
        if chunks:
            added = add_documents_to_index(chunks)
            total_added_chunks += added
            summary = {
                "filename": filename,
                "chunks": added,
                "uploaded_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            file_summaries.append(summary)
            manifest["uploaded_files"].append(summary)

    manifest["total_chunks"] = manifest.get("total_chunks", 0) + total_added_chunks
    save_manifest(manifest)

    return {
        "status": "success",
        "chunks_added": total_added_chunks,
        "files_processed": file_summaries,
        "total_chunks_in_index": manifest["total_chunks"]
    }

def get_index_stats() -> Dict[str, Any]:
    """Returns current manifest and index statistics."""
    manifest = load_manifest()
    index_exists = os.path.exists(FAISS_INDEX_PATH)
    return {
        "index_ready": index_exists,
        "total_chunks": manifest.get("total_chunks", 0),
        "uploaded_files": manifest.get("uploaded_files", [])
    }

def reset_index() -> Dict[str, Any]:
    """Resets the FAISS index to the baseline support dataset."""
    import shutil
    if os.path.exists(FAISS_INDEX_PATH):
        try:
            shutil.rmtree(FAISS_INDEX_PATH)
        except Exception as e:
            print(f"Error removing faiss index: {e}")
    ingest_data()
    return {"status": "success", "message": "Knowledge base index reset to default baseline."}

if __name__ == "__main__":
    ingest_data()
