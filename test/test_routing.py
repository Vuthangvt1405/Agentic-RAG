import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from router import dispatch

QUESTIONS = [
    # In-scope (theses: Zero Trust, ZTNA, micro-segmentation, SDN)
    "What is zero trust network access?",
    "How does micro-segmentation limit lateral movement?",
    "What are design principles for migrating to zero trust architecture?",
    # Out-of-scope
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "What is the current price of Bitcoin?",
]

for q in QUESTIONS:
    result = dispatch(q)
    print(f"{result['route']:12} <- {q}")
