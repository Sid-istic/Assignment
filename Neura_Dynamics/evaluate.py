"""
Evaluation Script for RAG System
Tests the system with predefined questions and measures performance.
"""

import time
from rag import generate_answer

# ============================================================================
# TEST QUESTIONS
# ============================================================================

QUESTIONS = [
    # Direct retrieval - simple fact
    "What is the time window for a full refund?",
    
    # Direct retrieval - pricing information
    "How much is expedited shipping?",
    
    # Synthesis - requires understanding multiple sections
    "Can I cancel my order if I already received it?",
    
    # Negative case - out of scope
    "Do you sell laptops?",
    
    # Edge case - specific scenario
    "I haven't received my refund after 10 days, what do I do?"
]

# ============================================================================
# EVALUATION FUNCTION
# ============================================================================

def run_evaluation():
    """
    Runs all test questions through the RAG system and displays results.
    """
    print("=" * 70)
    print("STARTING RAG SYSTEM EVALUATION")
    print("=" * 70)
    print()
    
    results = []
    
    for i, question in enumerate(QUESTIONS, 1):
        print(f"[{i}/{len(QUESTIONS)}] Q: {question}")
        
        # Measure response time
        start_time = time.time()
        answer = generate_answer(question)
        duration = time.time() - start_time
        
        # Display results
        print(f"     A: {answer}")
        print(f"     ⏱️  Time: {duration:.2f}s")
        print("-" * 70)
        print()
        
        # Store for potential further analysis
        results.append({
            "question": question,
            "answer": answer,
            "duration": duration
        })
    
    # Summary statistics
    avg_time = sum(r["duration"] for r in results) / len(results)
    print("=" * 70)
    print(f"✅ EVALUATION COMPLETE")
    print(f"   Total Questions: {len(results)}")
    print(f"   Average Response Time: {avg_time:.2f}s")
    print("=" * 70)
    
    return results

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    run_evaluation()
