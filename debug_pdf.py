#!/usr/bin/env python3
"""
Debug tool to test PDF extraction with your files.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

from utils.file_parser import FileParser

def test_pdf_file(file_path):
    """Test PDF extraction on a specific file."""
    print(f"🔍 Testing PDF extraction for: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    try:
        # Read file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Extract text
        extracted_text = FileParser.extract_text_from_pdf(file_content)
        
        # Validate
        is_valid = FileParser.validate_extracted_text(extracted_text)
        
        # Results
        print(f"📊 Extraction Results:")
        print(f"   • File size: {len(file_content)} bytes")
        print(f"   • Extracted text length: {len(extracted_text)} characters")
        print(f"   • Text validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        
        # Preview
        preview = FileParser.preview_text(extracted_text, 300)
        print(f"\n📖 Text Preview:")
        print("-" * 50)
        print(preview)
        print("-" * 50)
        
        if not is_valid:
            print("\n⚠️  Extraction issues detected:")
            if len(extracted_text) < 50:
                print("   • Text too short (less than 50 characters)")
            print("   • Try using 'Paste Text' method instead")
        
        return is_valid
        
    except Exception as e:
        print(f"❌ Error testing PDF: {e}")
        return False

def main():
    """Main debug function."""
    print("🧪 PDF Extraction Debug Tool")
    print("=" * 60)
    
    # Test common PDF locations
    test_files = [
        "New_Resume.pdf",
        "JD.pdf", 
        "resume.pdf",
        "job_description.pdf"
    ]
    
    found_files = []
    for file_name in test_files:
        if os.path.exists(file_name):
            found_files.append(file_name)
    
    if not found_files:
        print("📁 No PDF files found in current directory.")
        print("💡 To test your PDFs:")
        print("   1. Copy your PDF files to this folder")
        print("   2. Run: python debug_pdf.py")
        print("   3. Or specify path: python debug_pdf.py path/to/your/file.pdf")
        return
    
    # Test found files
    for file_path in found_files:
        test_pdf_file(file_path)
        print()
    
    print("=" * 60)
    print("💡 If extraction failed:")
    print("   • Use 'Paste Text' in the web interface")
    print("   • Copy text manually from your PDF viewer")
    print("   • Ensure PDFs contain selectable text (not scanned images)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Test specific file
        test_pdf_file(sys.argv[1])
    else:
        main()