"""
LangGraph State Machine - Shared LLM Instance

The orchestrator that connects all nodes into a self-correcting RAG Pipeline.

Flow:
Start -> Retrieve -> GraphAugment -> Generate -> Audit -> PASS/FAIL -> Loop from Retrieve(if FAIL) -> END  

Max retries prevent infinite loops.
"""
from typing import Literal
from langgraph.graph import StateGraph, END
from llama_cpp import Llama
from src.state import GraphState
from src.nodes.retrieve import RetrieveNode
from src.nodes.graph_augment import GraphAugmentNode
from src.nodes.generate import GenerateNode
from src.nodes.audit import AuditNode
from src.nodes.rewrite import RewriteNode
from config.paths import MISTRAL_GGUF_MODEL_PATH
from utils.gpu_selector import get_device
from utils.logger import get_logger
import traceback

logger = get_logger(__name__)

class CogitoGraph:
    """
    The complete Cogito RAG pipeline as a LangGraph state machine.

    This graph implements the self-correcting loop:
    - Retrieve relevant docs
    - Expand context with graph neigbors
    - Generate answer
    - Audit for hallucinations
    - If audit fails: rewrite query and retry
    - If audit passes: return answer
    
    Uses a SHARED LLM instance across all nodes to save memory.
    """
    def __init__(self, collection_name:str = "cogito_docs", max_retries: int=3):
        """
        Initialize the Cogito graph with shared LLM instance.
        Args:
            collection_name: ChromeDB collection name
            max_retries: Maximum retry attempts before giving up
        """
        self.max_retries = max_retries
        self.collection_name = collection_name

        logger.info("Initializing Cogito graph nodes...")
        
        # Load LLM ONCE (shared across all nodes)
        logger.info("Loading shared LLM instance...")
        backend, n_gpu_layers = get_device()
        logger.info(f"Auto-detected backend: {backend}")
        logger.info(f"GPU Layers: {n_gpu_layers}")
        
        try:
            import os
            model_path = str(MISTRAL_GGUF_MODEL_PATH)
            logger.info(f"Model path: {model_path}")
            logger.info(f"Model file exists: {os.path.exists(model_path)}")
            
            if os.path.exists(model_path):
                file_size = os.path.getsize(model_path) / (1024**3)  # GB
                logger.info(f"Model file size: {file_size:.2f} GB")
            else:
                raise FileNotFoundError(f"Model file not found at: {model_path}")
            
            logger.info(f"Attempting to load model with backend={backend}, n_gpu_layers={n_gpu_layers}")
            self.shared_llm = Llama(
                model_path=model_path,
                n_ctx=4096,
                n_gpu_layers=n_gpu_layers,
                verbose=True
            )
            logger.info("Shared LLM loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {type(e).__name__}: {str(e)}")
            logger.error(f"Model path attempted: {MISTRAL_GGUF_MODEL_PATH}")
            logger.error(f"Backend: {backend}, GPU layers: {n_gpu_layers}")
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            raise
        logger.info("✓ Shared LLM loaded successfully")

        # Initialize nodes (retrieval nodes don't need LLM)
        self.retrieve_node = RetrieveNode(collection_name=collection_name, top_k=10)
        self.graph_augment_node = GraphAugmentNode(collection_name=collection_name, depth=1, max_neighbors_per_node=3)
        
        # Pass the shared LLM to generation nodes
        self.generate_node = GenerateNode(llm=self.shared_llm, temperature=0.1, max_tokens=2048)
        self.audit_node = AuditNode(llm=self.shared_llm, temperature=0.0, max_tokens=150)
        self.rewrite_node = RewriteNode(llm=self.shared_llm, temperature=0.3, max_tokens=100)
        
        self.graph = self._build_graph()
        logger.info("✓ Cogito graph initialized")

    def _build_graph(self) -> StateGraph:
        """Build the langGraph state machine."""
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("graph_augment", self.graph_augment_node)
        workflow.add_node("generate", self.generate_node)
        workflow.add_node("audit", self.audit_node)
        workflow.add_node("rewrite", self.rewrite_node)

        # Define edges
        workflow.set_entry_point("retrieve")

        # Linear flow: retrieve -> graph_augment -> generate -> audit
        workflow.add_edge("retrieve", "graph_augment")
        workflow.add_edge("graph_augment", "generate")
        workflow.add_edge("generate","audit")

        # Conditional routing from Audit
        workflow.add_conditional_edges("audit", self._should_continue, 
                                        {
                                            "rewrite":"rewrite",  # If failed: rewrite query
                                            "end": END            # If passed: done 
                                        }
                                        )
        # Loop back: rewrite -> retrieve
        workflow.add_edge("rewrite", "retrieve")
        
        # Compile the graph
        return workflow.compile()

    def _should_continue(self, state: GraphState) -> Literal["rewrite","end"]:
        """
        Conditional routing logic after audit.
        Returns:
            "rewrite" if audit failed and retries remain
            "end" if audit passed or max retries reached
        """
        audit_status = state.audit_status
        retry_count = state.retry_count

        logger.info(f"\nDecision Point: Audit Status = {audit_status.upper()}")
        logger.info(f"Retry Count: {retry_count}/{self.max_retries}")

        if audit_status == "pass":
            logger.info("Audit PASSED - Ending Pipeline")
            return "end"
        
        if retry_count >= self.max_retries:
            logger.warning(f"Max Retries ({self.max_retries}) reached - Ending with failure")
            state.final_answer = f"Unable to provide a verified answer after {self.max_retries} attempts. Last attempt: {state.generation}"
            return "end"
        
        logger.info(f"Audit FAILED - Rewriting (attempt {retry_count + 1}/{self.max_retries})")
        return "rewrite"
    
    def query(self, question: str) -> dict:
        """Run a query through the pipeline"""
        from src.state import create_initial_state

        logger.info("COGITO PIPELINE STARTED")
        logger.info(f"Question: {question}")

        initial_state = create_initial_state(question)
        final_state = self.graph.invoke(initial_state)

        # Handle both dict and object state
        if isinstance(final_state, dict):
            response = {
                "question": question,
                "answer": final_state.get("final_answer") or final_state.get("generation", ""),
                "audit_status": final_state.get("audit_status", "unknown"),
                "audit_reason": final_state.get("audit_reason", ""),
                "retry_count": final_state.get("retry_count", 0),
                "sources": {
                    "vector_docs": len(final_state.get("documents", [])),
                    "graph_docs": len(final_state.get("graph_augmented_docs", []))
                }
            }
        else:
            response = {
                "question": question,
                "answer": final_state.final_answer or final_state.generation,
                "audit_status": final_state.audit_status,
                "audit_reason": final_state.audit_reason,
                "retry_count": final_state.retry_count,
                "sources": {
                    "vector_docs": len(final_state.documents),
                    "graph_docs": len(final_state.graph_augmented_docs)
                }
            }

        logger.info("PIPELINE COMPLETED")
        logger.info(f"Status: {response['audit_status'].upper()}")
        logger.info(f"Answer: {response['answer'][:100]}...")

        return response

    def run_with_updates(self, question: str):
        """
        Yields events as the graph executes.
        Used for providing progress updates to the UI.
        """
        inputs = {"question": question, "documents": [], "generation": "", "audit_status": "pending"}
        return self.graph.stream(inputs, stream_mode='updates')