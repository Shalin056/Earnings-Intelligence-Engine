"""
Phase 0.2: Install all required packages
"""

import subprocess
import sys

def install_requirements():
    """
    Installs all required packages from requirements.txt
    """
    print("="*60)
    print("INSTALLING REQUIREMENTS")
    print("="*60)
    print()
    
    try:
        # Upgrade pip first
        print("📦 Upgrading pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("✅ pip upgraded successfully\n")
        
        # Install requirements
        print("📦 Installing packages from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ All packages installed successfully!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Installation failed: {e}")
        return False

if __name__ == "__main__":
    success = install_requirements()
    
    if success:
        print("\n" + "="*60)
        print("✅ INSTALLATION COMPLETE")
        print("="*60)
        print("\nNext step: Run python verify_installation.py")
    else:
        print("\n" + "="*60)
        print("❌ INSTALLATION FAILED")
        print("="*60)
        print("\nPlease check the errors above and try again.")