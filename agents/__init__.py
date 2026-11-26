from typing import TypedDict, List, Optional

class AgentState(TypedDict, total=False):
    query: str
    intent: str
    documents: List[str]
    draft_answer: str
    is_hallucination: bool
    iterations: int
    provider: Optional[str]
    model: Optional[str]
    groq_api_key: Optional[str]
    openai_api_key: Optional[str]
