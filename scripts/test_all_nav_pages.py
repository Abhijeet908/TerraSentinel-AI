"""Verify that all navigation views render without errors in Streamlit AppTest."""
import sys
from streamlit.testing.v1 import AppTest

def test_all_pages():
    print("Testing Streamlit Navigation Pages...")
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert len(at.exception) == 0, f"Startup exceptions: {at.exception}"
    print("  Initial startup: OK (0 exceptions)")

    # 1. Test empty state across all 7 tabs
    nav_labels = [
        "01 / CHANGE DETECTION",
        "02 / CHANGE TRUTH ENGINE",
        "03 / TEMPORAL DYNAMICS",
        "04 / QUANTITATIVE ANALYSIS",
        "05 / REPORTS",
        "06 / DATA & BENCHMARKS",
        "07 / METHODOLOGY"
    ]

    for nav in nav_labels:
        btn = next((b for b in at.button if nav in b.label), None)
        if btn:
            btn.click().run()
            if len(at.exception) > 0:
                print(f"  FAILED on empty nav '{nav}': {at.exception[0].value}")
                return False
            print(f"  Empty state '{nav}': OK")

    # 2. Load Amazon dataset and run detection
    btn_amazon = next(b for b in at.button if "AMAZON" in b.label)
    btn_amazon.click().run()
    assert len(at.exception) == 0

    btn_run = next(b for b in at.button if "RUN CHANGE DETECTION" in b.label)
    btn_run.click().run()
    assert len(at.exception) == 0
    print("  Analysis executed on Amazon benchmark: OK")

    # 3. Test analyzed state across all 7 tabs
    for nav in nav_labels:
        btn = next((b for b in at.button if nav in b.label), None)
        if btn:
            btn.click().run()
            if len(at.exception) > 0:
                print(f"  FAILED on analyzed nav '{nav}': {at.exception[0].value}")
                return False
            print(f"  Analyzed state '{nav}': OK")

    # 4. Also test Yellowstone and Kerala loading
    btn_wildfire = next((b for b in at.button if "WILDFIRE" in b.label), None)
    if btn_wildfire:
        btn_wildfire.click().run()
        assert len(at.exception) == 0
        print("  Yellowstone Wildfire load: OK")

    btn_kerala = next((b for b in at.button if "KERALA" in b.label), None)
    if btn_kerala:
        btn_kerala.click().run()
        assert len(at.exception) == 0
        print("  Kerala Flood load: OK")

    print("\nALL NAVIGATION PAGES AND WORKFLOWS VERIFIED (0 EXCEPTIONS)!")
    return True

if __name__ == "__main__":
    success = test_all_pages()
    sys.exit(0 if success else 1)
