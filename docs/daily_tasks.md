# First 2 Weeks: Daily Tasks

**Goal:** Establish the foundational baseline RAG pipeline and fine-tune a custom embedding/intent model.

### Week 1: Data Ingestion & Baseline RAG (Focus: IR, Vector Search, RAG)
- **Day 1: Project Setup & Data Acquisition**
  - Initialize the repository, set up virtual environment, install dependencies (`requirements.txt`).
  - Download a public dataset (e.g., Ubuntu Dialogue Corpus or Amazon QA).
  - Write an exploratory script (`scripts/explore_data.py`) to understand and clean the raw data.
- **Day 2: Chunking Strategy**
  - Implement a basic chunking strategy using LangChain or LlamaIndex in a utility script.
  - Experiment with chunk sizes and overlap to optimize for support queries.
- **Day 3: Baseline Embeddings & Vector Store**
  - Generate embeddings using a pre-trained sentence-transformer (e.g., `all-MiniLM-L6-v2`).
  - Set up a FAISS index (`retrieval/vector_store.py`) and ingest the embedded chunks.
- **Day 4: Basic Retriever Setup**
  - Implement a baseline retriever module (`agents/retriever.py`) to query the FAISS index.
  - Test top-k retrieval manually with sample support queries.
- **Day 5: Baseline Generator & End-to-End Test**
  - Connect a basic LLM (via API or local small model) to generate answers based on retrieved context.
  - Create a simple script (`scripts/run_baseline.py`) to run the full RAG pipeline end-to-end.

### Week 2: Intent Classification & Custom Embeddings (Focus: PyTorch, PEFT, Representation Learning)
- **Day 6: Intent Dataset Preparation**
  - Label a subset of queries with basic intents (e.g., "technical issue", "billing", "product inquiry").
  - Prepare a dataset for fine-tuning a small classification model.
- **Day 7: LoRA Fine-Tuning Setup**
  - Set up a PyTorch + HuggingFace script (`scripts/train_intent.py`) to fine-tune DistilBERT using LoRA.
- **Day 8: Training the Classifier**
  - Train the intent classifier and log metrics (accuracy, loss) using Weights & Biases or MLflow.
  - Evaluate the classifier on a holdout set.
- **Day 9: Integrating Intent into Retrieval**
  - Modify the RAG pipeline to use the intent classifier to route or filter queries before retrieval.
- **Day 10: Week 1 & 2 Review and Cleanup**
  - Refactor code into the `agents`, `models`, and `retrieval` modules.
  - Write basic unit tests (`tests/test_retrieval.py`) for the ingestion and retrieval components.
