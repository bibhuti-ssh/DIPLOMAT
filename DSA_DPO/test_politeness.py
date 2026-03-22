"""
Independent test script for PolitenessScorer.
Run with: python3 test_politeness.py
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from politeness_scorer import PolitenessScorer

def main():
    print("="*50)
    print("Testing Politeness Scorer")
    print("="*50)

    print("\n[1] Initializing Scorer...")
    try:
        scorer = PolitenessScorer(use_convokit=True)
        print(f"✓ Initialized (Use ConvoKit: {scorer.use_convokit})")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return

    # Dummy test cases
    test_cases = [
        "Hello, I would verify the status if you could pass me the jar.", # Polite
        "Give me the jar now!", # Impolite
        "I understand your concern, but we must stick to the timeline.", # Neutral/Professional
        "", # Empty string (Edge case)
        "   ", # Whitespace (Edge case)
    ]

    print("\n[2] Running Test Cases...\n")
    
    for i, text in enumerate(test_cases):
        print(f"--- Case {i+1}: '{text}' ---")
        try:
            result = scorer.score_utterance(text)
            print(f"   -> Score: {result.score:.4f}")
            print(f"   -> Method: {result.method}")
            print(f"   -> Features: {result.features.keys() if result.features else 'None'}")
        except Exception as e:
            print(f"   -> ❌ FAILED: {e}")
        print("")

    if not scorer.use_convokit:
        print("\n⚠️  WARNING: ConvoKit was NOT used. Check initialization logs.")
    else:
        print("\n✅ ConvoKit appears to be active.")

if __name__ == "__main__":
    main()
