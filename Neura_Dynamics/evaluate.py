import time
from rag import generate_answer

QUESTIONS = [
    "What is the return policy window?",
    "Do you offer free shipping?",
    "Can I cancel my order after it has shipped?",
    "Do you sell laptops?",  # Should be "I don't know"
    "I haven't received my refund after 10 days, what do I do?"
]

def run_test():
    print("=== RAG System Evaluation ===")
    for q in QUESTIONS:
        print(f"\nQ: {q}")
        start = time.time()
        ans = generate_answer(q)
        end = time.time()
        print(f"A: {ans}")
        print(f"Time: {end - start:.2f}s")

if __name__ == "__main__":
    run_test()
