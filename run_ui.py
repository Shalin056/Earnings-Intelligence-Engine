"""
Launch Web UI for Earnings Call Intelligence Engine
"""

import subprocess
import sys
from pathlib import Path

def launch_ui():
    """
    Launch Streamlit web interface
    """
    print("\n" + "="*60)
    print("LAUNCHING WEB INTERFACE")
    print("="*60)
    print()
    print("🚀 Starting Earnings Call Intelligence Engine UI...")
    print()
    print("📊 Features:")
    print("   • Real-time Dashboard")
    print("   • Interactive Analysis")
    print("   • Agent Monitoring")
    print("   • Results Explorer")
    print("   • Model Comparison")
    print("   • Configuration Panel")
    print()
    print("🌐 The UI will open in your default browser")
    print("   URL: http://localhost:8501")
    print()
    print("⏹️  Press Ctrl+C to stop the server")
    print()
    print("="*60)
    print()
    
    # Get ui path
    ui_path = Path(__file__).parent / 'ui' / 'app.py'
    
    if not ui_path.exists():
        print("❌ Error: UI application not found")
        print(f"   Looking for: {ui_path}")
        return False
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(ui_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ])
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped")
        print("👋 Thank you for using the Earnings Call Intelligence Engine!")
    except Exception as e:
        print(f"❌ Error launching UI: {e}")
        return False
    
    return True


if __name__ == "__main__":
    launch_ui()