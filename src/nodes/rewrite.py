"""
Rewrite Node - Query Refinement

When the AuditNode detects Hallucinations, this node reformulates the question
to retrieve better context on the next loop iteration.

Example:
- Original: "What is cudaMalloc?"
- Audit fails: "Parameters are unsupported"
- Rewrite: "What are the parameters and return values of cudaMalloc?"
- Retry retrieval with refined query
"""
import sys
from typing import Optional
from llama_cpp import Llama 
from src.state import GraphState
from config.paths import MISTRAL_GGUF_MODEL_PATH
from utils.logger import get_logger
from utils.custom_exception import CustomException 

logger = get_logger(__name__)

class RewriteNode:
    """
    Query rewriting node

    Input: state.question, state.audit_reason, state.generation
    Output: state.question (updated with refined query)

    Analyze why the audit failed and reformulates the question to
    target the missing information.
    """

    def __init__(self,
                model_path: Optional[str] = None,
                n_ctx: int = 4096,
                n_gpu_layers: int = -1,
                temperature: float = 0.3,
                max_tokens: int = 100):
        """
        Initialize the rewrite node.

        Args:
            model_path: Path to GGUF model
            n_ctx: Context window size
            n_gpu_layers: GPU layers (-1 = all)
            temperature: Low temp for focused rewriting
            max_tokens: Short rewrite needed
        """
        self.model_path = model_path or str(MISTRAL_GGUF_MODEL_PATH)
        self.temperature = temperature
        self.max_tokens = max_tokens
        logger.info(f"Loading Rewrite LLM from {self.model_path}")

        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx = n_ctx,
                n_gpu_layers = n_gpu_layers,
                verbose = False)
            logger.info("Rewrite LLM loaded successfully")
        except Exception as e:
            raise CustomException(f"Failed to load Rewrite LLM: {e}", sys)

    def _build_rewrite_prompt(self,
                            original_question: str,
                            failed_answer: str,
                            audit_reason: str) -> str:
        """
        Build the query rewriting prompt.
        This analyzes what went wrong and reformulates the question.
        """
        prompt = f"""You are a query refinement assistant. A documentation search failed to find accurate information.
        
        Original Question:{original_question}
        
        Attempted Answer (which failed verification):{failed_answer}

        Why it failed:{audit_reason}

        Task: Rewrite the question to be more specific and target the missing information.
        Focus on technical details, parameters, or context that were lacking.

        Examples:
        - Original: "What is cudaMalloc?"
        - Failed because: "Parameters were not mentioned in sources"
        - Rewrite: "What are the parameters and return type of cudaMalloc?"

        - Original: "How does authentication work?"
        - Failed because: "Version-specific details missing"
        - Rewrite: "What are the authentication parameters for API v2.0?"

        Rewritten Question (one sentence, specific, technical):"""

        return prompt
    
    def __call__(self, state:GraphState) -> GraphState:
        """
        Execute query rewriting.
        Arg:
            state: Current graph state
        Returns:
            Updated state with refined question
        """
        try:
            original_question = state.question
            failed_answer = state.generation
            audit_reason = state.audit_reason
            retry_count = state.retry_count

            logger.info(f"Rewriting query (attempt {retry_count + 1})...")
            logger.info(f"Original: {original_question}")
            logger.info(f"Audit failure reason: {audit_reason}")

            prompt = self._build_rewrite_prompt(original_question, failed_answer, audit_reason)
            logger.info("Generating refined query...")
            response = self.llm(prompt,
                                max_tokens=self.max_tokens,
                                temperature =self.temperature,
                                stop = ["\n\n","ORIGINAL:","REWRITTEN:"],
                                echo=False)
            # Extract rewritten question
            rewritten = response['choices'][0]['text'].strip()
            #Cleanup Common artifacts
            rewritten = rewritten.replace("REWRITTEN QUESTION:", "").strip()
            rewritten = rewritten.replace("Rewrite:", "").strip()
            rewritten = rewritten.strip('"\'')

            if not rewritten or len(rewritten)<10:
                logger.warning("Rewrite too short, using fallback strategy")
                rewritten = f"{original_question} (with complete details and parameters)"

            logger.info(f"Rewritten: {rewritten}")
            
            # Update state
            state.question = rewritten
            state.retry_count += 1

            # Clear previous generation to force new retrieval
            state.documents = []
            state.document_ids = []
            state.graph_augmented_docs = []
            state.generation = ""
            state.audit_status = "pending"
            
            return state

        except Exception as e:
            raise CustomException(f"Query rewriting failed: {e}", sys)
    
    def rewrite_node(state: GraphState) -> GraphState:
        """
        Standalone rewrite function for LangGraph.
        Args:
            state: Current graph state
        Returns:
            Updated state with rewritten question
        """
        rewriter = RewriteNode()
        return rewriter(state)