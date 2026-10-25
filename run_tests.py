"""Master Test Runner for TerraSentinel AI."""
import unittest
import sys
import os

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from tests.test_upgrade_suite import TestTerraSentinelUpgrade

def run_master_test_suite():
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestTerraSentinelUpgrade)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    print("\n" + "="*70)
    if result.wasSuccessful():
        print(f"SUCCESS: ALL {result.testsRun} COMPETITIVE SPECIFICATION TESTS PASSED!")
        print("="*70)
        return 0
    else:
        print(f"FAILURE: {len(result.failures) + len(result.errors)} TEST(S) FAILED.")
        print("="*70)
        return 1

if __name__ == "__main__":
    exit_code = run_master_test_suite()
    sys.exit(exit_code)
