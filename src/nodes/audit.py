"""
Audit Node - Hallucination Detection

This node acts as a "fact-checker" that verifies whether the generated
answer is actually supported by the retriever context. If not, it triggers
a rewrite and retry loop.

This prevents hallucinations in technical documentations.
"""
import sys
from typing import Optional
from llama_cpp import Llama 
from src.state import GraphState
from config.paths import MISTRAL_GGUF_MODEL_PATH
from utils.logger import get_logger
from utils.custom_exception import CustomException 

logger = get_logger(__name__)

class AuditNode:
    """
    Hallucination detection node.

    Input: state.generation, state.documents, state.graph_augmented_docs
    Output: state.audit_status, state.audit_reason

    Uses a small, fast LLM to verify if the generated answer is 
    factually grounded in the source documents.
    """

    def __init__(self,
                llm: Optional[Llama] = None,
                model_path: Optional[str] = None,
                n_ctx: int = 4096,
                n_gpu_layers: int = -1,
                temperature: float = 0.0, # Zero temp for deterministic yes/no
                max_tokens: int=100):
        """
        Initialize the audit node.

        Args:
            llm: Optional shared LLM instance
            model_path: Path to GGUF model (only used if llm is None)
            n_ctx: Context window size
            n_gpu_layers: GPU layers (-1 = all)
            temperature: 0.0 for deterministic verification
            max_tokens: Short response needed (just yes/no + reason)
        """
        self.temperature = temperature
        self.max_tokens = max_tokens

        if llm is not None:
            # Use shared LLM instance
            self.llm = llm
            logger.info("Using shared LLM instance for audit")
        else:
            # Load new LLM instance
            self.model_path = model_path or str(MISTRAL_GGUF_MODEL_PATH)
            logger.info(f"Loading Audit LLM from {self.model_path}")

            try:
                self.llm = Llama(model_path=self.model_path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)
                logger.info("Audit LLM loaded successfully")
            
            except Exception as e:
                raise CustomException(f"Failed to load Audit LLM: {e}", sys)
        
    
    def _build_audit_prompt(self, answer: str, documents: list, graph_docs: list) -> str:
        """
        Build the audit verification prompt.
        This is the CRITICAL prompt that determines accuracy.
        """
        # Combine all source documents
        all_docs = documents + graph_docs
        context = "\n\n".join([f"[Source {i+1}]\n{doc}" for i, doc in enumerate(all_docs)])

        # Audit Prompt
        prompt = f"""You are a fact-checking AI. Your job is to verify if an answer is FULLY supported by the provided source documents.
        
        Source Documents: {context}

        Answer to Verify: {answer}
        
        Task: Determine if the answer above is completely supported by the source documents.
        
        Rules:
        1. Every factual claim in the answer must be found in the sources
        2. If the answer add information NOT in the sources, it fails
        3. If the answer contradicts the sources, it fails
        4. If sources are incomplete/unclear for the answer, it fails
        5. If all facts match sources, it pass

        Respond ONLY with:
        PASS - [reason why it passes]
        OR
        FAIL - [what claims are unsupported]

        Your verdict:"""
        return prompt

    def _parse_audit_response(self, response: str) -> tuple:
        """
        Parse the audit LLM's response.
        Returns:
            (verdict, reason) where verdict is "pass" or "fail"
        """
        response_lower = response.lower().strip()
        verdict = "fail" # default
        reason = response.strip()

        if 'verdict:' in response_lower:
            verdict_line = response_lower.split("verdict:")[1].split("\n")[0].strip()
            if "pass" in verdict_line and "fail" not in verdict_line:
                verdict = "pass"
            elif "fail" in verdict_line:
                verdict = "fail"
        
        elif "pass" in response_lower and "fail" not in response_lower:
            verdict = "pass"
        elif "fail" in response_lower:
            verdict = "fail"

        elif any(word in response_lower for word in ["yes", "correct", "accurate", "supported", "verified"]):
            verdict = "pass"
        elif any(word in response_lower for word in ["no", "incorrect", "inaccurate", "contradicts", "unverified", "missing"]):
            verdict = "fail"

        if "reason:" in response_lower:
            reason_parts = response.split("REASON:")
            if len(reason_parts) > 1:
                reason = reason_parts[-1].strip()
            else:
                reason_parts = response.split("reason:")
                if len(reason_parts) > 1:
                    reason = reason_parts[-1].strip()
        
        reason = reason.split("\n")[0].strip()
        if not reason or len(reason)<15:
            reason = response[:200]
        
        logger.info(f"Parsed verdict: {verdict.upper()}, reason: {reason[:100]}")
        return verdict, reason


    def __call__(self, state: GraphState) -> GraphState:
        """
        Execute audit check.

        Args:
            state: Current graph state
        
        Returns:
            Updated state with audit_status and audit_reason
        """
        try:
            generation = state.generation
            documents = state.documents
            graph_docs = state.graph_augmented_docs

            logger.info("Starting audit check...")
            logger.info(f"Checking {len(generation)} chars against {len(documents)} + {len(graph_docs)} docs")

            # Check if we have content to audit
            if not generation or generation.strip() == "":
                logger.warning("Empty generation, marking as FAIL")
                state.audit_status = "FAIL"
                state.audit_reason = "No answer was generated"
                return state
            
            # Build audit prompt
            prompt = self._build_audit_prompt(generation, documents, graph_docs)

            # Run audit
            logger.info("Running audit LLM...")
            response = self.llm(prompt, max_tokens=self.max_tokens, temperature=self.temperature, stop=["\n\n","VERDICT:","###"], echo=False)

            # Extract audit result
            audit_text = response['choices'][0]['text'].strip()
            logger.info(f"Audit response: {audit_text[:200]}")

            # Parse verdict
            verdict, reason = self._parse_audit_response(audit_text)

            # Update state
            state.audit_status = verdict
            state.audit_reason = reason

            if verdict == "pass":
                logger.info(f"AUDIT PASSED: {reason}")
                state.final_answer = generation
            else:
                logger.warning(f"AUDIT FAILED: {reason}")
            
            return state
        
        except Exception as e:
            raise CustomException(f"Audit Failed: {e}", sys)

def audit_node(state: GraphState) -> GraphState:
    """
    Standalone audit function for LangGraph.

    Args:
        state: Current graph state
    Returns:
        Updated state with audit results
    """
    auditor = AuditNode()
    return auditor(state)