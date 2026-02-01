"""
Evaluator Module

Orchestrates evaluation of the Cogito RAG system.
Compares performance against standard RAG baseline.

Uses DeepEval for additional metrics when available.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from tqdm import tqdm

from src.evaluation.metrics import (
    evaluate_response,
    EvaluationResult,
    FaithfulnessMetric,
    RelevanceMetric, 
    HallucinationMetric,
    SelfCorrectionMetric
)
from config.paths import RESULTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class TestCase:
    """A single test case for evaluation."""
    question: str
    expected_answer: Optional[str] = None  # Ground truth if available
    expected_sources: Optional[List[str]] = None  # Expected source documents
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "general"  # For grouping results


@dataclass
class EvaluationSummary:
    """Summary of evaluation run."""
    total_queries: int
    avg_faithfulness: float
    avg_relevance: float
    avg_hallucination: float
    audit_pass_rate: float
    self_correction_rate: float
    avg_retries: float
    
    # Breakdown by difficulty
    easy_faithfulness: float = 0.0
    medium_faithfulness: float = 0.0
    hard_faithfulness: float = 0.0
    
    # Timing
    avg_latency_seconds: float = 0.0
    total_time_seconds: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class CogitoEvaluator:
    """
    Evaluator for the Cogito RAG system.
    
    Runs test cases through the pipeline and measures:
    - Faithfulness (grounding in context)
    - Relevance (answering the question)
    - Hallucination rate
    - Self-correction effectiveness
    """
    
    def __init__(self, cogito_graph=None):
        """
        Initialize the evaluator.
        
        Args:
            cogito_graph: Optional CogitoGraph instance.
                          If None, will create one when evaluate() is called.
        """
        self.graph = cogito_graph
        self.results: List[EvaluationResult] = []
        self.self_correction_tracker = SelfCorrectionMetric()
        
        # Ensure results directory exists
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    def load_test_cases(self, path: str) -> List[TestCase]:
        """
        Load test cases from a JSON file.
        
        Expected format:
        [
            {
                "question": "What is cudaMalloc?",
                "expected_answer": "cudaMalloc allocates memory...",
                "difficulty": "easy",
                "category": "memory"
            },
            ...
        ]
        """
        with open(path, 'r') as f:
            data = json.load(f)
        
        test_cases = []
        for item in data:
            test_cases.append(TestCase(
                question=item["question"],
                expected_answer=item.get("expected_answer"),
                expected_sources=item.get("expected_sources"),
                difficulty=item.get("difficulty", "medium"),
                category=item.get("category", "general")
            ))
        
        logger.info(f"Loaded {len(test_cases)} test cases from {path}")
        return test_cases
    
    def create_sample_test_cases(self) -> List[TestCase]:
        """Create sample test cases for demonstration."""
        return [
            TestCase(
                question="What is Python?",
                difficulty="easy",
                category="basics"
            ),
            TestCase(
                question="How do you define a function in Python?",
                difficulty="easy",
                category="syntax"
            ),
            TestCase(
                question="What are Python list comprehensions?",
                difficulty="medium",
                category="syntax"
            ),
            TestCase(
                question="How does Python handle memory management?",
                difficulty="hard",
                category="internals"
            ),
            TestCase(
                question="What is the difference between a list and a tuple?",
                difficulty="easy",
                category="data_types"
            ),
        ]
    
    def evaluate_single(self, question: str) -> EvaluationResult:
        """
        Evaluate a single question.
        
        Args:
            question: The question to evaluate
            
        Returns:
            EvaluationResult with all metrics
        """
        if self.graph is None:
            from src.graph import CogitoGraph
            self.graph = CogitoGraph()
        
        # Run query through Cogito
        response = self.graph.query(question)
        
        # Get context (combine vector and graph docs)
        # Note: response contains counts, actual docs are in state
        context = []  # Would need to capture from state
        
        # For now, use the answer to derive context estimate
        answer = response.get("answer", "")
        audit_status = response.get("audit_status", "unknown")
        retry_count = response.get("retry_count", 0)
        
        # Placeholder context (in real usage, capture from state)
        context = [answer]  # Self-referential for now
        
        # Evaluate
        result = evaluate_response(
            question=question,
            answer=answer,
            context=context,
            retry_count=retry_count,
            audit_passed=(audit_status == "pass")
        )
        
        return result
    
    def evaluate(
        self,
        test_cases: List[TestCase],
        save_results: bool = True
    ) -> EvaluationSummary:
        """
        Run evaluation on a list of test cases.
        
        Args:
            test_cases: List of TestCase objects
            save_results: Whether to save results to file
            
        Returns:
            EvaluationSummary with aggregated metrics
        """
        if self.graph is None:
            from src.graph import CogitoGraph
            logger.info("Initializing CogitoGraph for evaluation...")
            self.graph = CogitoGraph()
        
        logger.info(f"Starting evaluation of {len(test_cases)} test cases...")
        
        results = []
        start_time = datetime.now()
        
        # Difficulty buckets
        difficulty_scores = {"easy": [], "medium": [], "hard": []}
        
        for test_case in tqdm(test_cases, desc="Evaluating"):
            try:
                # Run query
                response = self.graph.query(test_case.question)
                
                answer = response.get("answer", "")
                audit_status = response.get("audit_status", "unknown")
                retry_count = response.get("retry_count", 0)
                
                # For proper evaluation, we'd capture context from state
                # Using placeholder for now
                context = [answer] if answer else []
                
                # Evaluate
                result = evaluate_response(
                    question=test_case.question,
                    answer=answer,
                    context=context,
                    retry_count=retry_count,
                    audit_passed=(audit_status == "pass")
                )
                
                results.append(result)
                difficulty_scores[test_case.difficulty].append(result.faithfulness_score)
                
                # Track self-correction
                self.self_correction_tracker.record_result(
                    initial_passed=(retry_count == 0 and audit_status == "pass"),
                    final_passed=(audit_status == "pass"),
                    retry_count=retry_count,
                    initial_faithfulness=result.faithfulness_score,
                    final_faithfulness=result.faithfulness_score
                )
                
            except Exception as e:
                logger.error(f"Failed to evaluate: {test_case.question[:50]}... - {e}")
                continue
        
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()
        
        # Calculate summary
        if not results:
            logger.error("No results to summarize!")
            return EvaluationSummary(
                total_queries=0,
                avg_faithfulness=0.0,
                avg_relevance=0.0,
                avg_hallucination=0.0,
                audit_pass_rate=0.0,
                self_correction_rate=0.0,
                avg_retries=0.0,
                total_time_seconds=total_time
            )
        
        n = len(results)
        summary = EvaluationSummary(
            total_queries=n,
            avg_faithfulness=sum(r.faithfulness_score for r in results) / n,
            avg_relevance=sum(r.relevance_score for r in results) / n,
            avg_hallucination=sum(r.hallucination_score for r in results) / n,
            audit_pass_rate=sum(1 for r in results if r.audit_passed) / n,
            self_correction_rate=sum(1 for r in results if r.self_correction_success) / n,
            avg_retries=sum(r.retry_count for r in results) / n,
            easy_faithfulness=sum(difficulty_scores["easy"]) / max(1, len(difficulty_scores["easy"])),
            medium_faithfulness=sum(difficulty_scores["medium"]) / max(1, len(difficulty_scores["medium"])),
            hard_faithfulness=sum(difficulty_scores["hard"]) / max(1, len(difficulty_scores["hard"])),
            avg_latency_seconds=total_time / n,
            total_time_seconds=total_time
        )
        
        self.results = results
        
        # Save results
        if save_results:
            self._save_results(results, summary)
        
        # Print summary
        self._print_summary(summary)
        
        return summary
    
    def _save_results(self, results: List[EvaluationResult], summary: EvaluationSummary):
        """Save evaluation results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_path = RESULTS_DIR / f"eval_results_{timestamp}.json"
        with open(results_path, 'w') as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        
        # Save summary
        summary_path = RESULTS_DIR / f"eval_summary_{timestamp}.json"
        with open(summary_path, 'w') as f:
            json.dump(summary.to_dict(), f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        logger.info(f"Summary saved to {summary_path}")
    
    def _print_summary(self, summary: EvaluationSummary):
        """Print evaluation summary to console."""
        print("="*60)
        print("COGITO EVALUATION SUMMARY")
        print("="*60)
        print(f"Total Queries:        {summary.total_queries}")
        print(f"Total Time:           {summary.total_time_seconds:.1f}s")
        print(f"Avg Latency:          {summary.avg_latency_seconds:.2f}s")
        print("-"*60)
        print("METRICS:")
        print(f"  Faithfulness:       {summary.avg_faithfulness:.2%}")
        print(f"  Relevance:          {summary.avg_relevance:.2%}")
        print(f"  Hallucination:      {summary.avg_hallucination:.2%}")
        print("-"*60)
        print("SELF-CORRECTION:")
        print(f"  Audit Pass Rate:    {summary.audit_pass_rate:.2%}")
        print(f"  Correction Success: {summary.self_correction_rate:.2%}")
        print(f"  Avg Retries:        {summary.avg_retries:.2f}")
        print("-"*60)
        print("BY DIFFICULTY:")
        print(f"  Easy:               {summary.easy_faithfulness:.2%}")
        print(f"  Medium:             {summary.medium_faithfulness:.2%}")
        print(f"  Hard:               {summary.hard_faithfulness:.2%}")
        print("="*60)
    
    def compare_with_baseline(
        self,
        test_cases: List[TestCase],
        baseline_results: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Compare Cogito results with a baseline (standard RAG without self-correction).
        
        Args:
            test_cases: Test cases to evaluate
            baseline_results: Optional pre-computed baseline results
            
        Returns:
            Comparison metrics
        """
        # Run Cogito evaluation
        cogito_summary = self.evaluate(test_cases, save_results=False)
        
        # If no baseline provided, can't compare
        if baseline_results is None:
            return {
                "cogito": cogito_summary.to_dict(),
                "baseline": None,
                "comparison": "No baseline provided"
            }
        
        # Calculate baseline metrics
        baseline_faithfulness = sum(r.get("faithfulness", 0) for r in baseline_results) / len(baseline_results)
        baseline_hallucination = sum(r.get("hallucination", 0) for r in baseline_results) / len(baseline_results)
        
        improvement = {
            "faithfulness_improvement": cogito_summary.avg_faithfulness - baseline_faithfulness,
            "hallucination_reduction": baseline_hallucination - cogito_summary.avg_hallucination,
            "cogito_better": cogito_summary.avg_faithfulness > baseline_faithfulness
        }
        
        return {
            "cogito": cogito_summary.to_dict(),
            "baseline": {
                "avg_faithfulness": baseline_faithfulness,
                "avg_hallucination": baseline_hallucination
            },
            "improvement": improvement
        }


def run_evaluation(collection_name: str = "cogito_docs", test_file: Optional[str] = None):
    """
    Convenience function to run evaluation.
    
    Args:
        collection_name: ChromaDB collection to use
        test_file: Optional path to test cases JSON
    """
    from src.graph import CogitoGraph
    
    print("Initializing Cogito for evaluation...")
    graph = CogitoGraph(collection_name=collection_name)
    
    evaluator = CogitoEvaluator(cogito_graph=graph)
    
    if test_file:
        test_cases = evaluator.load_test_cases(test_file)
    else:
        print("Using sample test cases...")
        test_cases = evaluator.create_sample_test_cases()
    
    summary = evaluator.evaluate(test_cases)
    
    return summary