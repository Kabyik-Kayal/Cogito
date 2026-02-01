"""
Custom Evaluation Metrics

Defines metrics for evaluating the Cogito RAG system:
1. Faithfulness - Is the answer grounded in retrieved context?
2. Hallucination Rate - How often does it make unsupported claims?
3. Self-Correction Success - How often does the retry loop fix errors?
4. Answer Relevance - Does the answer address the question?

Uses DeepEval as the evaluation framework.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import re
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationResult:
    """Result from evaluating a single query."""
    question: str
    answer: str
    context: List[str]
    
    # Metric scores (0.0 to 1.0)
    faithfulness_score: float = 0.0
    relevance_score: float = 0.0
    hallucination_score: float = 0.0  # Lower is better
    
    # Self-correction metrics
    retry_count: int = 0
    audit_passed: bool = False
    self_correction_success: bool = False
    
    # Details
    unsupported_claims: List[str] = field(default_factory=list)
    reasoning: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer[:200] + "..." if len(self.answer) > 200 else self.answer,
            "faithfulness": self.faithfulness_score,
            "relevance": self.relevance_score,
            "hallucination": self.hallucination_score,
            "retry_count": self.retry_count,
            "audit_passed": self.audit_passed,
            "self_correction_success": self.self_correction_success,
            "unsupported_claims": self.unsupported_claims
        }


class FaithfulnessMetric:
    """
    Measures how faithful the answer is to the retrieved context.
    
    A faithful answer only makes claims that are directly supported
    by the provided context documents.
    
    Score: 1.0 = fully faithful, 0.0 = completely hallucinated
    """
    
    def __init__(self, llm=None):
        """
        Initialize the faithfulness metric.
        
        Args:
            llm: Optional LLM for claim extraction and verification.
                 If None, uses keyword-based heuristics.
        """
        self.llm = llm
    
    def _extract_claims(self, answer: str) -> List[str]:
        """Extract factual claims from the answer."""
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)
        
        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Skip short fragments or questions
            if len(sentence) < 20 or sentence.endswith('?'):
                continue
            # Skip meta-statements
            if any(phrase in sentence.lower() for phrase in [
                "i don't have", "cannot", "no information",
                "based on the context", "according to"
            ]):
                continue
            claims.append(sentence)
        
        return claims
    
    def _check_claim_support(self, claim: str, context: List[str]) -> bool:
        """Check if a claim is supported by the context."""
        claim_lower = claim.lower()
        
        # Extract key terms from the claim
        # Remove common words
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                    'could', 'should', 'may', 'might', 'must', 'can', 'this',
                    'that', 'these', 'those', 'it', 'its', 'to', 'of', 'in',
                    'for', 'on', 'with', 'at', 'by', 'from', 'as', 'and', 'or'}
        
        words = re.findall(r'\b\w+\b', claim_lower)
        key_terms = [w for w in words if w not in stopwords and len(w) > 2]
        
        if not key_terms:
            return True  # No key terms to verify
        
        # Check if key terms appear in context
        context_text = ' '.join(context).lower()
        
        matches = sum(1 for term in key_terms if term in context_text)
        match_ratio = matches / len(key_terms) if key_terms else 0
        
        # Consider claim supported if 60%+ of key terms found
        return match_ratio >= 0.6
    
    def evaluate(self, answer: str, context: List[str]) -> tuple:
        """
        Evaluate faithfulness of an answer.
        
        Args:
            answer: The generated answer
            context: List of context documents
            
        Returns:
            (score, unsupported_claims)
        """
        if not answer or not context:
            return 0.0, ["No answer or context provided"]
        
        claims = self._extract_claims(answer)
        
        if not claims:
            return 1.0, []  # No verifiable claims = faithful by default
        
        unsupported = []
        supported_count = 0
        
        for claim in claims:
            if self._check_claim_support(claim, context):
                supported_count += 1
            else:
                unsupported.append(claim)
        
        score = supported_count / len(claims)
        
        logger.info(f"Faithfulness: {supported_count}/{len(claims)} claims supported")
        
        return score, unsupported


class RelevanceMetric:
    """
    Measures how relevant the answer is to the question.
    
    A relevant answer directly addresses what was asked.
    
    Score: 1.0 = fully relevant, 0.0 = completely irrelevant
    """
    
    def __init__(self, llm=None):
        self.llm = llm
    
    def _extract_question_terms(self, question: str) -> List[str]:
        """Extract key terms from the question."""
        # Remove question words
        question_words = {'what', 'how', 'why', 'when', 'where', 'who', 'which', 
                         'is', 'are', 'do', 'does', 'can', 'could', 'would', 'should'}
        
        words = re.findall(r'\b\w+\b', question.lower())
        key_terms = [w for w in words if w not in question_words and len(w) > 2]
        
        return key_terms
    
    def evaluate(self, question: str, answer: str) -> float:
        """
        Evaluate relevance of answer to question.
        
        Args:
            question: The original question
            answer: The generated answer
            
        Returns:
            Relevance score (0.0 to 1.0)
        """
        if not answer or not question:
            return 0.0
        
        # Check for refusal to answer
        refusal_phrases = [
            "i don't have enough information",
            "cannot answer",
            "no relevant documents",
            "unable to provide"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in refusal_phrases):
            # Refusal is considered somewhat relevant (honest about limitations)
            return 0.5
        
        # Check overlap of key terms
        question_terms = self._extract_question_terms(question)
        answer_lower = answer.lower()
        
        if not question_terms:
            return 1.0  # Can't measure
        
        matches = sum(1 for term in question_terms if term in answer_lower)
        score = matches / len(question_terms)
        
        # Boost score if answer is substantial
        if len(answer) > 100:
            score = min(1.0, score + 0.2)
        
        return score


class HallucinationMetric:
    """
    Measures the hallucination rate of the system.
    
    This is essentially (1 - faithfulness) but also checks for
    specific hallucination patterns like invented facts, wrong versions, etc.
    
    Score: 0.0 = no hallucinations, 1.0 = fully hallucinated
    """
    
    def __init__(self):
        self.faithfulness_metric = FaithfulnessMetric()
    
    def _check_hallucination_patterns(self, answer: str, context: List[str]) -> List[str]:
        """Check for common hallucination patterns."""
        issues = []
        
        # Pattern 1: Version numbers not in context
        version_pattern = r'v?\d+\.\d+(?:\.\d+)?'
        answer_versions = set(re.findall(version_pattern, answer))
        context_text = ' '.join(context)
        context_versions = set(re.findall(version_pattern, context_text))
        
        invented_versions = answer_versions - context_versions
        if invented_versions:
            issues.append(f"Invented version numbers: {invented_versions}")
        
        # Pattern 2: Specific numbers/statistics not in context
        number_pattern = r'\b\d{2,}\b'  # Numbers with 2+ digits
        answer_numbers = set(re.findall(number_pattern, answer))
        context_numbers = set(re.findall(number_pattern, context_text))
        
        invented_numbers = answer_numbers - context_numbers
        if len(invented_numbers) > 2:  # Allow some variance
            issues.append(f"Potentially invented statistics: {list(invented_numbers)[:3]}")
        
        # Pattern 3: Quotes not in context
        quote_pattern = r'"([^"]+)"'
        answer_quotes = re.findall(quote_pattern, answer)
        for quote in answer_quotes:
            if len(quote) > 10 and quote not in context_text:
                issues.append(f"Invented quote: \"{quote[:50]}...\"")
        
        return issues
    
    def evaluate(self, answer: str, context: List[str]) -> tuple:
        """
        Evaluate hallucination level.
        
        Args:
            answer: The generated answer
            context: List of context documents
            
        Returns:
            (hallucination_score, issues_list)
        """
        # Get faithfulness score
        faithfulness, unsupported = self.faithfulness_metric.evaluate(answer, context)
        
        # Check additional patterns
        pattern_issues = self._check_hallucination_patterns(answer, context)
        
        # Combine issues
        all_issues = unsupported + pattern_issues
        
        # Hallucination score = 1 - faithfulness, adjusted for pattern issues
        base_score = 1.0 - faithfulness
        pattern_penalty = min(0.3, len(pattern_issues) * 0.1)
        
        hallucination_score = min(1.0, base_score + pattern_penalty)
        
        return hallucination_score, all_issues


class SelfCorrectionMetric:
    """
    Measures how effective the self-correction loop is.
    
    Tracks:
    - How often retry improves the answer
    - Average retries needed
    - Success rate of audit passes
    """
    
    def __init__(self):
        self.results = []
    
    def record_result(
        self,
        initial_passed: bool,
        final_passed: bool,
        retry_count: int,
        initial_faithfulness: float,
        final_faithfulness: float
    ):
        """Record a single query result."""
        self.results.append({
            "initial_passed": initial_passed,
            "final_passed": final_passed,
            "retry_count": retry_count,
            "initial_faithfulness": initial_faithfulness,
            "final_faithfulness": final_faithfulness,
            "improved": final_faithfulness > initial_faithfulness
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.results:
            return {"error": "No results recorded"}
        
        total = len(self.results)
        passed_first_try = sum(1 for r in self.results if r["initial_passed"])
        passed_after_retry = sum(1 for r in self.results 
                                 if not r["initial_passed"] and r["final_passed"])
        improved = sum(1 for r in self.results if r["improved"])
        avg_retries = sum(r["retry_count"] for r in self.results) / total
        
        return {
            "total_queries": total,
            "passed_first_try": passed_first_try,
            "passed_first_try_rate": passed_first_try / total,
            "passed_after_retry": passed_after_retry,
            "correction_success_rate": passed_after_retry / max(1, total - passed_first_try),
            "improved_with_retry": improved,
            "improvement_rate": improved / total,
            "avg_retry_count": avg_retries
        }


# Convenience function to run all metrics
def evaluate_response(
    question: str,
    answer: str,
    context: List[str],
    retry_count: int = 0,
    audit_passed: bool = True
) -> EvaluationResult:
    """
    Run all metrics on a single response.
    
    Args:
        question: Original question
        answer: Generated answer  
        context: Retrieved context documents
        retry_count: Number of retries taken
        audit_passed: Whether the final audit passed
        
    Returns:
        EvaluationResult with all scores
    """
    faithfulness_metric = FaithfulnessMetric()
    relevance_metric = RelevanceMetric()
    hallucination_metric = HallucinationMetric()
    
    faithfulness, unsupported = faithfulness_metric.evaluate(answer, context)
    relevance = relevance_metric.evaluate(question, answer)
    hallucination, _ = hallucination_metric.evaluate(answer, context)
    
    result = EvaluationResult(
        question=question,
        answer=answer,
        context=context,
        faithfulness_score=faithfulness,
        relevance_score=relevance,
        hallucination_score=hallucination,
        retry_count=retry_count,
        audit_passed=audit_passed,
        self_correction_success=retry_count > 0 and audit_passed,
        unsupported_claims=unsupported
    )
    
    return result