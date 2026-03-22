"""
Quick test script to verify all pipeline fixes are working correctly.
Run this before running the full pipeline.
"""

import os
import sys


def test_gcns_classifier_loading():
    """Test that GcNS classifier loads strategies from strategy_mapping.json"""
    print("\n" + "=" * 60)
    print("TEST 1: GcNS Classifier Strategy Loading")
    print("=" * 60)

    try:
        from strategy_classifier import load_trained_gcns_classifier

        model_path = "trained_models/gcns_negotiation_classifier"
        classifier = load_trained_gcns_classifier(model_path)

        print(f"✅ Classifier loaded successfully")
        print(f"✅ Strategies loaded: {len(classifier.strategy_labels)}")
        print(f"✅ Strategy labels: {', '.join(classifier.strategy_labels[:3])}...")

        # Verify expected strategies
        expected_strategies = [
            "Active Listening",
            "Anchoring",
            "Collaborative Style",
            "Credibility Assertion",
            "Data-Driven Justification",
        ]

        for strat in expected_strategies:
            if strat in classifier.strategy_labels:
                print(f"   ✓ Found: {strat}")
            else:
                print(f"   ✗ Missing: {strat}")
                return False

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_strategy_normalization():
    """Test strategy normalization function"""
    print("\n" + "=" * 60)
    print("TEST 2: Strategy Normalization")
    print("=" * 60)

    try:
        from session_scorer import normalize_strategy_for_alignment

        # Mock alignment matrix
        mock_matrix = {
            ("Collaborative Style / Win-Win Framing", "Active Listening"): 1.0,
            ("Principled Negotiation / Data-Driven Justification", "Anchoring"): 0.5,
        }

        # Test normalization
        test_cases = [
            ("Collaborative Style", "Collaborative Style / Win-Win Framing"),
            ("Win-Win Framing", "Collaborative Style / Win-Win Framing"),
            (
                "Principled Negotiation",
                "Principled Negotiation / Data-Driven Justification",
            ),
            ("Active Listening", "Active Listening"),
        ]

        for input_strat, expected_output in test_cases:
            result = normalize_strategy_for_alignment(input_strat, mock_matrix)
            if result == expected_output:
                print(f"   ✓ {input_strat} → {result}")
            else:
                print(f"   ✗ {input_strat} → {result} (expected {expected_output})")
                return False

        print("✅ Strategy normalization working correctly")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_alignment_matrix_loading():
    """Test alignment matrix loads correctly"""
    print("\n" + "=" * 60)
    print("TEST 3: Alignment Matrix Loading")
    print("=" * 60)

    try:
        from session_scorer import load_alignment_matrix

        matrix_path = "alignment_matrices/negotiation_alignment.csv"
        if not os.path.exists(matrix_path):
            print(f"⚠️  Matrix file not found: {matrix_path}")
            return True  # Not critical for test

        matrix = load_alignment_matrix(matrix_path)

        print(f"✅ Alignment matrix loaded")
        print(f"✅ Total strategy pairs: {len(matrix)}")

        # Show a few examples
        count = 0
        for (c_strat, e_strat), score in matrix.items():
            if count < 3:
                print(f"   ({c_strat}, {e_strat}) → {score}")
                count += 1

        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_imports():
    """Test all critical imports work"""
    print("\n" + "=" * 60)
    print("TEST 4: Critical Imports")
    print("=" * 60)

    imports = [
        ("strategy_classifier", "load_trained_gcns_classifier"),
        ("strategy_classifier", "GcNSClassifier"),
        ("session_scorer", "SessionScorer"),
        ("session_scorer", "normalize_strategy_for_alignment"),
        ("llm_judge", "LLMJudge"),
        ("session_generator", "SessionGenerator"),
    ]

    all_success = True
    for module_name, obj_name in imports:
        try:
            module = __import__(module_name, fromlist=[obj_name])
            obj = getattr(module, obj_name)
            print(f"   ✓ {module_name}.{obj_name}")
        except Exception as e:
            print(f"   ✗ {module_name}.{obj_name}: {e}")
            all_success = False

    if all_success:
        print("✅ All imports successful")
    return all_success


def test_config_yaml():
    """Test config.yaml has Gemini section"""
    print("\n" + "=" * 60)
    print("TEST 5: Config YAML")
    print("=" * 60)

    try:
        import yaml

        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Check for key sections
        required_sections = ["weights", "llm_judge", "self_play", "gemini"]

        for section in required_sections:
            if section in config:
                print(f"   ✓ Section '{section}' found")
            else:
                print(f"   ✗ Section '{section}' missing")
                return False

        # Check Gemini config
        if "default_model" in config["gemini"]:
            print(f"   ✓ Gemini default_model: {config['gemini']['default_model']}")

        print("✅ Config YAML valid")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def main():
    print("\n" + "=" * 70)
    print("DSA-DPO PIPELINE - FIX VERIFICATION TESTS")
    print("=" * 70)

    # Change to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Run tests
    tests = [
        ("Imports", test_imports),
        ("Config YAML", test_config_yaml),
        ("GcNS Classifier Loading", test_gcns_classifier_loading),
        ("Strategy Normalization", test_strategy_normalization),
        ("Alignment Matrix Loading", test_alignment_matrix_loading),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:10} {test_name}")

    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Pipeline fixes verified.")
        print("\nNext steps:")
        print("  1. Run: python phase1_negative_generation.py --test")
        print(
            "  2. Or with Gemini: python phase1_negative_generation.py --test --model gemini-2.5-flash"
        )
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please review errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
