"""
State Definition for LangGraph

This module defines the state structure that flows through all nodes
in the RAG pipeline. The state is a typed dictionary that gets passed
between nodes and modified at each step.
"""

from typing import TypedDict, List, Dict, Any, Literal
from pydantic import BaseModel, Field

class Document(BaseModel):
    """
    A document chunk with metadata.

    Used internally by retrieval nodes before being added to state.

    Attributes:
        content: The text content of the document.
        node_id: The NetworkX graph node ID (used for graph expansion)
        metadata: Additional metadata (source, section, etc.)
    """
    content: str
    node_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def __str__(self):
        return f"Document(node_id={self.node_id}, content_length={len(self.content)})"

class GraphState(BaseModel):
    """
    The  state object that flows through the LangGraph pipeline.

    Each node reads from and writes to this state object.
    This is the 'memory' that persists across the entire RAG loop.

    Attributes:
        question: The Original user query
        documents: Retrieved document chunks (as strings)
        document_ids: IDs of retrieved documents (for graph expansion)
        graph_augmented_docs: Additional documents from graph neighbors
        generation: The LLM's generated answer (draft)
        audit_status: Result of audit check ("pass","fail", or "needs_correction")
        audit_reason: Explanation of why audit failed (if applicable)
        retry_count: Number of times the loop has retried
        final_answer: The verified final answer (only set on success)
    """
    question: str
    documents: List[str] = Field(default_factory=list)
    document_ids: List[str] = Field(default_factory=list)
    graph_augmented_docs: List[str] = Field(default_factory=list)
    generation: str = ""
    audit_status: Literal["pass", "fail", "needs_correction", "pending"] = "pending"
    audit_reason: str = ""
    retry_count: int = 0
    final_answer: str = ""

    class Config:
        # Allow arbitrary types for compatibility with LangGraph
        arbitrary_types_allowed = True

def create_initial_state(question: str) -> GraphState:
    """
    Create a fresh state for a new query.

    Args:
        question: The user's question
    
    Returns:
        Initial GraphState with empty/default values
    """
    return GraphState(question=question)