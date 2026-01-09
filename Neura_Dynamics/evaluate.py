from rag import generate_answer
import time

QUESTIONS = [
    # 1. Direct Retrieval
    "What is the time window for a full refund?",
    # 2. Direct Retrieval
    "How much is expedited shipping?",
    # 3. Synthesis / Multiple docs (maybe)
    "Can I cancel my order if I already received it?", 
    # 4. Negative / Out of scope
    "Do you sell laptops?",
    # 5. Edge case
    "I haven't received my refund after 10 days, what do I do?"
]

def run_evaluation():
    print("Starting Evaluation...\n")
    results = []
    
    for q in QUESTIONS:
        print(f"Q: {q}")
        start_time = time.time()
        answer = generate_answer(q)
        duration = time.time() - start_time
        
        print(f"A: {answer}")
        print(f"[Time: {duration:.2f}s]\n" + "-"*40 + "\n")
        
        results.append({
            "question": q,
            "answer": answer,
            "duration": duration
        })

if __name__ == "__main__":
    run_evaluation()
