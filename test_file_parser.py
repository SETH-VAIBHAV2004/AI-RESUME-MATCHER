#!/usr/bin/env python3
"""
Test script for file parser functionality.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_file_parser():
    """Test the file parser utility."""
    print("🧪 Testing File Parser...")
    
    try:
        from utils.file_parser import FileParser
        
        # Test text validation
        valid_text = "This is a sample resume with enough content to be considered valid for parsing and analysis."
        invalid_text = "Short"
        
        assert FileParser.validate_extracted_text(valid_text), "Should validate good text"
        assert not FileParser.validate_extracted_text(invalid_text), "Should reject short text"
        
        # Test text preview
        long_text = "This is a very long text that should be truncated when creating a preview. " * 10
        preview = FileParser.preview_text(long_text, 100)
        assert len(preview) <= 103, "Preview should be truncated"  # 100 + "..."
        
        print("  ✅ File parser utility works correctly")
        return True
        
    except ImportError as e:
        print(f"  ❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"  ❌ File parser test failed: {e}")
        return False

def main():
    """Run file parser tests."""
    print("🧪 File Parser Tests")
    print("=" * 40)
    
    success = test_file_parser()
    
    print("=" * 40)
    if success:
        print("🎉 File parser is ready!")
        print("📄 You can now upload PDF, DOCX, and TXT files in the web interface.")
    else:
        print("❌ File parser tests failed.")
    
    return success

if __name__ == "__main__":
    main()