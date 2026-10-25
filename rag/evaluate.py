import os
from dotenv import load_dotenv

from rag.retriever import Retriever

load_dotenv()

# (query, playbook section that must appear in top-k)
TESTS = [
    ("Month-to-month contract, short tenure, fiber optic", "Month-to-month contract customers"),
    ("New customer in first 3 months, tenure 2 months", "Short tenure"),
    ("Fiber optic internet, speed complaints", "Fiber optic customers"),
    ("Pays by electronic check, billing issues", "Electronic check payers"),
    ("No tech support, customer struggles with setup", "No tech support"),
    ("High monthly charges, expensive plan", "High monthly charges"),
]

K = 3

r = Retriever(os.getenv("GEMINI_API_KEY"))
hits = 0
for query, expected in TESTS:
    results = r.search(query, k=K)
    titles = [c.split("\n")[0] for c in results]
    ok = any(expected.lower() in t.lower() for t in titles)
    hits += ok
    print(f"{'PASS' if ok else 'FAIL'} | {query}\n      expected: {expected}\n      got: {titles}\n")

print(f"Hit rate @{K}: {hits}/{len(TESTS)} = {hits/len(TESTS):.0%}")