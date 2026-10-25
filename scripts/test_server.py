"""Test script to verify Streamlit app startup and response."""
import subprocess
import time
import urllib.request
import sys

def main():
    print("Starting Streamlit process...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless=true", "--server.port=8501"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    try:
        # Wait up to 10 seconds for server to respond
        success = False
        for attempt in range(12):
            time.sleep(1)
            try:
                with urllib.request.urlopen("http://localhost:8501", timeout=2) as resp:
                    print(f"HTTP response code: {resp.status}")
                    if resp.status == 200:
                        content = resp.read().decode('utf-8', errors='ignore')
                        if "TerraSentinel" in content or "streamlit" in content.lower():
                            print("Streamlit app served HTML successfully!")
                            success = True
                            break
            except Exception as e:
                # Still starting up
                pass

        if not success:
            stdout, stderr = proc.communicate(timeout=3)
            print("Server stdout:\n", stdout)
            print("Server stderr:\n", stderr)
            raise RuntimeError("Streamlit did not respond with 200 OK")

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Streamlit server process terminated.")

if __name__ == "__main__":
    main()
