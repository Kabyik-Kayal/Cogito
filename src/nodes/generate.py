"""
Generate Node

Uses a local LLM to draft an answer based on retrieved documents.

This node runs Mistral GGUF model using llama-cpp-python.
It generates an initial answer that will be checked by the AuditNode.
"""
import sys
from typing import Optional
from llama_cpp import Llama
from src.state import GraphState
from config.paths import MISTRAL_GGUF_MODEL_PATH
from utils.gpu_selector import get_device
from utils.logger import get_logger
from utils.custom_exception import CustomException


logger = get_logger(__name__)

class GenerateNode:
    """
    LLM generation node.

    Input: state.question, state.documents, state.graph
    Output: state.generation

    Uses llama-cpp-python for inference
    """
    def __init__(self,
        model_path: Optional[str] = None,
        n_ctx: int = 4096,
        n_gpu_layers: Optional[int]=None,
        temperature: float = 0.1,
        max_tokens: int = 512):

        self.model_path = model_path or str(MISTRAL_GGUF_MODEL_PATH)
        self.temperature = temperature
        self.max_tokens = max_tokens

        backend, n_gpu_layers = get_device()
        logger.info(f"Auto-detected backend: {backend}")
        logger.info(f"GPU Layers: {n_gpu_layers}")
        logger.info(f"Loading LLM from {self.model_path}")
        
        try:
            self.llm = Llama(
                model_path = self.model_path,
                n_ctx = n_ctx,
                n_gpu_layers = n_gpu_layers,
                verbose = True
            )
            logger.info("LLM loaded successfully")
        
        except Exception as e:
            raise CustomException(f"Failed to load LLM: {e}", sys)
        
    def _build_prompt(
        self,
        question: str,
        documents: str,
        graph_docs: list
    ) -> str:
        """
        Build the RAG prompt with all context.
        Combines vector-retrieved docs + graph-augmented docs.
        """
        
        # Combine all context
        all_docs = document + graph_docs
        
        # Build context string
        context = "\n\n".join([f"[Document {i+1}]\n{doc}" for i, doc in enumerate(all_docs)])

        # RAG prompt template
        prompt = f"""You are a technical documentation assistant.
        Answer the question based ONLY on the provided context.
        If the context doesn't contain enough information to answer accurately,
        say "I don't have enough information to answer this question."

        Context: {context}
        Question: {question}

        Answer: Provide a clear, accurate answer based solely on the context above. Do not add information not present in the context.
        """
        return prompt

    def __call__(self, state: GraphState) -> GraphState:
        """
        Execute generation.
        Args:
            state: Current graph state
        Returns:
            Updated state with generation populated
        """
        try:
            question = state.question
            documents = state.documents
            graph_docs = state.graph_augmented_docs

            logger.info(f"Generating answer for: '{question}'")
            logger.info(f"Context: {len(documents)} vector docs + {len(graph_docs)} graph docs")

            # Check if we have any context
            if not documents and not graph_docs:
                logger.warning("No context available for generation!")
                state.generation = "No relevant documents found to answer this question"
                return state

            # Build prompt
            prompt = self._build_prompt(question, documents, graph_docs)

            # Generate response
            logger.info("Generating response (this may take several minutes)")
            response = self.llm(prompt, max_tokens=self.max_tokens, temperature=self.temperature, stop=["Question:", "\n\n\n"], echo=False)

            # Extract generated text
            generated_text = response['choices'][0]['text'].strip()
            logger.info(f"Generated {len(generated_text)} characters")
            logger.info(f"Priview: {generated_text[:100]}...")

            # Update state
            state.generation = generated_text

            return state
        
        except Exception as e:
            raise CustomException(f"Generation failed: {e}", sys)

def generate_node(state: GraphState) -> GraphState:
    """
    Standalone generation function for LangGraph.

    Args:
        state: Current graph state
    
    Returns:
        Updated state with generated answer
    """
    generator = GenerateNode()
    return generator(state)
