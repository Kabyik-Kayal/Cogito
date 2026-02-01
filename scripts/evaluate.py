#!/usr/bin/env python3
"""
Evaluation Script

CLI entry point for running evaluation on the Cogito system.

Usage:
    # Run with sample test cases
    python scripts/evaluate.py
    
    # Run with custom test file
    python scripts/evaluate.py --test-file tests/cuda_questions.json
    
    # Specify collection
    python scripts/evaluate.py --collection my_docs --test-file tests/questions.json
"""

import argparse
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.evaluation.evaluator import CogitoEvaluator, TestCase, run_evaluation
from config.paths import RESULTS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def create_test_file_template():
    """Create a template test file for users to fill in."""
    template = [
        {
            "question": "What is the syntax for defining a function?",
            "expected_answer": "Functions are defined using the 'def' keyword...",
            "difficulty": "easy",
            "category": "syntax"
        },
        {
            "question": "How do you handle memory allocation?",
            "expected_answer": None,
            "difficulty": "medium",
            "category": "memory"
        },
        {
            "question": "Explain the difference between X and Y with version-specific details.",
            "expected_answer": None,
            "difficulty": "hard",
            "category": "advanced"
        }
    ]
    
    template_path = RESULTS_DIR / "test_cases_template.json"
    with open(template_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    print(f"Template created: {template_path}")
    return template_path


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation on the Cogito RAG system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with sample questions
  python scripts/evaluate.py
  
  # Run with custom test file
  python scripts/evaluate.py --test-file tests/my_questions.json
  
  # Create template for custom tests
  python scripts/evaluate.py --create-template
        """
    )
    
    parser.add_argument(
        "--collection",
        type=str,
        default="cogito_docs",
        help="ChromaDB collection name (default: cogito_docs)"
    )
    
    parser.add_argument(
        "--test-file",
        type=str,
        help="Path to JSON file with test cases"
    )
    
    parser.add_argument(
        "--create-template",
        action="store_true",
        help="Create a template test file and exit"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to evaluate"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    # Create template if requested
    if args.create_template:
        create_test_file_template()
        return 0
    
    print("COGITO EVALUATION")
    
    try:
        # Initialize evaluator
        from src.graph import CogitoGraph
        
        print(f"Loading CogitoGraph (collection: {args.collection})...")
        graph = CogitoGraph(collection_name=args.collection)
        
        evaluator = CogitoEvaluator(cogito_graph=graph)
        
        # Load or create test cases
        if args.test_file:
            print(f"Loading test cases from {args.test_file}...")
            test_cases = evaluator.load_test_cases(args.test_file)
        else:
            print("Using sample test cases...")
            test_cases = evaluator.create_sample_test_cases()
        
        # Limit if requested
        if args.max_questions:
            test_cases = test_cases[:args.max_questions]
        
        print(f"Evaluating {len(test_cases)} questions...\n")
        
        # Run evaluation
        summary = evaluator.evaluate(
            test_cases=test_cases,
            save_results=not args.no_save
        )
        
        print("Evaluation complete!")
        
        # Return code based on performance
        if summary.avg_faithfulness >= 0.7:
            print("Status: GOOD - Faithfulness >= 70%")
            return 0
        elif summary.avg_faithfulness >= 0.5:
            print("Status: NEEDS IMPROVEMENT - Faithfulness 50-70%")
            return 0
        else:
            print("Status: POOR - Faithfulness < 50%")
            return 1
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
