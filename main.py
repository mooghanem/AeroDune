import os
import subprocess
import sys

def main():
    """
    Main application launcher. Starts the Streamlit dashboard server.
    """
    print("=" * 60)
    print(" 🏜️  Starting AeroDune Physics & Economic Framework")
    print("=" * 60)
    
    dashboard_path = os.path.join(os.path.dirname(__file__), "app", "dashboard.py")
    
    if not os.path.exists(dashboard_path):
        print(f"Error: Could not locate dashboard entry file at '{dashboard_path}'")
        sys.exit(1)
        
    cmd = [sys.executable, "-m", "streamlit", "run", dashboard_path]
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nShutting down AeroDune framework gracefully.")

if __name__ == "__main__":
    main()
