#!/usr/bin/env python3
"""
Setup script for Resume-Job Description Matcher
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error during {description.lower()}: {e}")
        if e.stdout:
            print(f"Output: {e.stdout}")
        if e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_requirements():
    """Install Python requirements."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    command = f"{sys.executable} -m pip install -r {requirements_file}"
    return run_command(command, "Installing Python dependencies")

def download_spacy_model():
    """Download spaCy English model."""
    command = f"{sys.executable} -m spacy download en_core_web_sm"
    return run_command(command, "Downloading spaCy English model")

def test_imports():
    """Test if all required modules can be imported."""
    print("🔄 Testing module imports...")
    
    required_modules = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'spacy',
        'gensim',
        'transformers',
        'torch',
        'rapidfuzz',
        'matplotlib',
        'plotly'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"❌ Failed to import: {', '.join(failed_imports)}")
        return False
    
    print("✅ All required modules imported successfully")
    return True

def create_sample_files():
    """Ensure sample files exist."""
    print("🔄 Checking sample files...")
    
    data_dir = Path(__file__).parent / "data"
    
    required_files = [
        "skills_dict.json",
        "sample_dataset.csv",
        "sample_resume.txt",
        "sample_job.txt"
    ]
    
    missing_files = []
    for file_name in required_files:
        file_path = data_dir / file_name
        if not file_path.exists():
            missing_files.append(file_name)
        else:
            print(f"  ✅ {file_name}")
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All sample files present")
    return True

def main():
    """Main setup function."""
    print("🚀 Resume-Job Description Matcher Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please install manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("⚠️  spaCy model download failed. It will be downloaded automatically on first run.")
    
    # Test imports
    if not test_imports():
        print("❌ Some modules failed to import. Please check your installation.")
        sys.exit(1)
    
    # Check sample files
    if not create_sample_files():
        print("❌ Some sample files are missing. Please ensure all files are present.")
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("   1. Launch web app: python main.py --web")
    print("   2. CLI analysis: python main.py --resume data/sample_resume.txt --job data/sample_job.txt")
    print("   3. Direct Streamlit: streamlit run app/app.py")
    print("\n💡 For help: python main.py --help")
    print("=" * 50)

if __name__ == "__main__":
    main()