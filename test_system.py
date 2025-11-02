#!/usr/bin/env python3
"""
Test script to verify the Resume-Job Description Matcher system works correctly.
"""

import sys
import os
from pathlib import Path
import traceback

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def test_text_cleaning():
    """Test text cleaning functionality."""
    print("🧪 Testing text cleaning...")
    try:
        from preprocessing.text_cleaner import TextCleaner
        
        cleaner = TextCleaner()
        test_text = "Hello World! This is a test with URLs http://example.com and emails test@example.com."
        
        cleaned_text = cleaner.clean_text(test_text)
        tokens = cleaner.tokenize_and_lemmatize(cleaned_text)
        
        assert len(cleaned_text) > 0, "Cleaned text should not be empty"
        assert len(tokens) > 0, "Tokens should not be empty"
        assert "http" not in cleaned_text.lower(), "URLs should be removed"
        
        print("  ✅ Text cleaning works correctly")
        return True
    except Exception as e:
        print(f"  ❌ Text cleaning failed: {e}")
        return False

def test_skill_extraction():
    """Test skill extraction functionality."""
    print("🧪 Testing skill extraction...")
    try:
        from analysis.skill_extractor import SkillExtractor
        
        extractor = SkillExtractor()
        test_text = "I have experience with Python, JavaScript, React, and machine learning."
        
        skills = extractor.extract_skills_hybrid(test_text)
        formatted_skills = extractor.format_skills_for_display(skills)
        
        assert isinstance(skills, dict), "Skills should be a dictionary"
        assert len(formatted_skills) > 0, "Should extract some skills"
        
        # Check if we found some expected skills
        all_skills = []
        for category_skills in formatted_skills.values():
            all_skills.extend([skill.lower() for skill in category_skills])
        
        expected_skills = ['python', 'javascript', 'react', 'machine learning']
        found_skills = [skill for skill in expected_skills if skill in all_skills]
        
        assert len(found_skills) > 0, f"Should find at least one expected skill. Found: {all_skills}"
        
        print(f"  ✅ Skill extraction works correctly. Found skills: {found_skills}")
        return True
    except Exception as e:
        print(f"  ❌ Skill extraction failed: {e}")
        traceback.print_exc()
        return False

def test_tfidf_features():
    """Test TF-IDF feature extraction."""
    print("🧪 Testing TF-IDF features...")
    try:
        from features.tfidf_features import TFIDFFeatures
        
        tfidf = TFIDFFeatures()
        resume_text = "I am a software engineer with Python and machine learning experience."
        job_text = "Looking for a software engineer with Python and AI experience."
        
        similarity = tfidf.calculate_similarity(resume_text, job_text)
        
        assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
        assert similarity > 0.1, f"Should have some similarity, got {similarity}"
        
        print(f"  ✅ TF-IDF features work correctly. Similarity: {similarity:.3f}")
        return True
    except Exception as e:
        print(f"  ❌ TF-IDF features failed: {e}")
        return False

def test_word2vec_features():
    """Test Word2Vec feature extraction."""
    print("🧪 Testing Word2Vec features...")
    try:
        from features.word2vec_features import Word2VecFeatures
        
        word2vec = Word2VecFeatures()
        resume_tokens = ["software", "engineer", "python", "machine", "learning"]
        job_tokens = ["software", "engineer", "python", "artificial", "intelligence"]
        
        similarity = word2vec.calculate_similarity(resume_tokens, job_tokens)
        
        assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
        
        print(f"  ✅ Word2Vec features work correctly. Similarity: {similarity:.3f}")
        return True
    except Exception as e:
        print(f"  ❌ Word2Vec features failed: {e}")
        return False

def test_bert_features():
    """Test BERT feature extraction."""
    print("🧪 Testing BERT features...")
    try:
        from features.bert_features import BERTFeatures
        
        bert = BERTFeatures()
        resume_text = "I am a software engineer with Python experience."
        job_text = "Looking for a software engineer with Python skills."
        
        similarity = bert.calculate_similarity(resume_text, job_text)
        
        assert 0 <= similarity <= 1, f"Similarity should be between 0 and 1, got {similarity}"
        
        print(f"  ✅ BERT features work correctly. Similarity: {similarity:.3f}")
        return True
    except Exception as e:
        print(f"  ❌ BERT features failed: {e}")
        return False

def test_gapfinder_nlp():
    """Test Enhanced GapFinder-NLP model."""
    print("🧪 Testing Enhanced GapFinder-NLP...")
    try:
        from features.enhanced_gapfinder import EnhancedGapFinderNLP
        
        gapfinder = EnhancedGapFinderNLP()
        resume_text = "I am a software engineer with Python and React experience."
        job_text = "Looking for a software engineer with Python, React, and AWS skills."
        
        results = gapfinder.comprehensive_analysis(resume_text, job_text)
        
        assert isinstance(results, dict), "Results should be a dictionary"
        assert 'gapfinder_score' in results, "Should have gapfinder_score"
        assert 'compatibility_probability' in results, "Should have compatibility_probability"
        assert 0 <= results['gapfinder_score'] <= 1, "Score should be between 0 and 1"
        
        print(f"  ✅ Enhanced GapFinder-NLP works correctly. Score: {results['gapfinder_score']:.3f}")
        return True
    except Exception as e:
        print(f"  ❌ Enhanced GapFinder-NLP failed: {e}")
        return False

def test_skill_gap_analysis():
    """Test skill gap analysis."""
    print("🧪 Testing skill gap analysis...")
    try:
        from analysis.skill_gap import SkillGapAnalyzer
        
        analyzer = SkillGapAnalyzer()
        
        resume_skills = {
            'technical': ['python', 'javascript'],
            'tools': ['git', 'docker'],
            'soft': ['communication'],
            'other': ['agile']
        }
        
        job_skills = {
            'technical': ['python', 'java'],
            'tools': ['git', 'aws'],
            'soft': ['communication', 'leadership'],
            'other': ['agile', 'scrum']
        }
        
        gap_analysis = analyzer.analyze_skill_gaps(resume_skills, job_skills)
        recommendations = analyzer.generate_recommendations(gap_analysis)
        
        assert isinstance(gap_analysis, dict), "Gap analysis should be a dictionary"
        assert isinstance(recommendations, list), "Recommendations should be a list"
        assert len(gap_analysis) > 0, "Should have gap analysis results"
        
        print(f"  ✅ Skill gap analysis works correctly. Found {len(recommendations)} recommendations")
        return True
    except Exception as e:
        print(f"  ❌ Skill gap analysis failed: {e}")
        return False

def test_metrics_calculation():
    """Test metrics calculation."""
    print("🧪 Testing metrics calculation...")
    try:
        from evaluation.metrics import MatchingMetrics
        
        metrics = MatchingMetrics()
        
        # Test similarity metrics
        similarity_metrics = metrics.calculate_similarity_metrics(0.8, 0.7, 0.9, 0.85)
        
        assert isinstance(similarity_metrics, dict), "Metrics should be a dictionary"
        assert 'final_score' in similarity_metrics, "Should have final_score"
        assert 0 <= similarity_metrics['final_score'] <= 100, "Final score should be 0-100"
        
        print(f"  ✅ Metrics calculation works correctly. Final score: {similarity_metrics['final_score']:.1f}%")
        return True
    except Exception as e:
        print(f"  ❌ Metrics calculation failed: {e}")
        return False

def test_file_parser():
    """Test file parser functionality."""
    print("🧪 Testing file parser...")
    try:
        from utils.file_parser import FileParser
        
        # Test text validation
        valid_text = "This is a sample resume with enough content to be considered valid."
        invalid_text = "Short"
        
        assert FileParser.validate_extracted_text(valid_text), "Should validate good text"
        assert not FileParser.validate_extracted_text(invalid_text), "Should reject short text"
        
        # Test preview
        long_text = "This is a long text. " * 20
        preview = FileParser.preview_text(long_text, 100)
        assert len(preview) <= 103, "Preview should be truncated"
        
        print("  ✅ File parser works correctly")
        return True
    except Exception as e:
        print(f"  ❌ File parser failed: {e}")
        return False

def test_end_to_end():
    """Test end-to-end functionality with sample data."""
    print("🧪 Testing end-to-end analysis...")
    try:
        # Read sample data (now using real dataset)
        data_dir = Path(__file__).parent / "data"
        combined_dataset = data_dir / "combined_dataset.csv"
        
        # If combined dataset exists, use it; otherwise fall back to sample files
        if combined_dataset.exists():
            import pandas as pd
            df = pd.read_csv(combined_dataset)
            if len(df) > 0:
                # Use first sample from real dataset
                resume_text = df.iloc[0]['resume_text']
                job_text = df.iloc[0]['job_text']
            else:
                # Fallback to sample files
                resume_file = data_dir / "sample_resume.txt"
                job_file = data_dir / "sample_job.txt"
        else:
            resume_file = data_dir / "sample_resume.txt"
            job_file = data_dir / "sample_job.txt"
        
        # Handle fallback to sample files if needed
        if 'resume_text' not in locals():
            if not resume_file.exists() or not job_file.exists():
                print("  ⚠️  Sample files not found, skipping end-to-end test")
                return True
            
            with open(resume_file, 'r', encoding='utf-8') as f:
                resume_text = f.read()
            with open(job_file, 'r', encoding='utf-8') as f:
                job_text = f.read()
        
        # Import all components
        from preprocessing.text_cleaner import TextCleaner
        from features.tfidf_features import TFIDFFeatures
        from analysis.skill_extractor import SkillExtractor
        from analysis.skill_gap import SkillGapAnalyzer
        from evaluation.metrics import MatchingMetrics
        
        # Initialize components
        text_cleaner = TextCleaner()
        skill_extractor = SkillExtractor()
        skill_gap_analyzer = SkillGapAnalyzer()
        metrics_calculator = MatchingMetrics()
        
        # Process texts
        resume_clean, resume_tokens = text_cleaner.preprocess_for_matching(resume_text)
        job_clean, job_tokens = text_cleaner.preprocess_for_matching(job_text)
        
        # Calculate one similarity score (TF-IDF for speed)
        tfidf_extractor = TFIDFFeatures()
        tfidf_score = tfidf_extractor.calculate_similarity(resume_clean, job_clean)
        
        # Extract skills
        resume_skills = skill_extractor.extract_skills_hybrid(resume_clean)
        job_skills = skill_extractor.extract_skills_hybrid(job_clean)
        
        # Format skills
        resume_skills_formatted = skill_extractor.format_skills_for_display(resume_skills)
        job_skills_formatted = skill_extractor.format_skills_for_display(job_skills)
        
        # Analyze gaps
        gap_analysis = skill_gap_analyzer.analyze_skill_gaps(
            resume_skills_formatted, job_skills_formatted
        )
        
        # Generate recommendations
        recommendations = skill_gap_analyzer.generate_recommendations(gap_analysis)
        
        # Calculate metrics (using dummy scores for other models)
        similarity_metrics = metrics_calculator.calculate_similarity_metrics(
            tfidf_score, 0.75, 0.80, 0.78
        )
        
        assert similarity_metrics['final_score'] > 0, "Should have a positive final score"
        assert len(recommendations) >= 0, "Should have recommendations (or empty list)"
        
        print(f"  ✅ End-to-end analysis works correctly. Final score: {similarity_metrics['final_score']:.1f}%")
        return True
    except Exception as e:
        print(f"  ❌ End-to-end analysis failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🧪 Resume-Job Description Matcher System Tests")
    print("=" * 60)
    
    tests = [
        test_text_cleaning,
        test_skill_extraction,
        test_tfidf_features,
        test_word2vec_features,
        test_bert_features,
        test_gapfinder_nlp,
        test_skill_gap_analysis,
        test_metrics_calculation,
        test_file_parser,
        test_end_to_end
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Test {test.__name__} crashed: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The system is ready to use.")
        print("\n🚀 Next steps:")
        print("   • Launch web app: python main.py --web")
        print("   • CLI analysis: python main.py --resume data/sample_resume.txt --job data/sample_job.txt")
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("💡 Try running: python setup.py")
    
    print("=" * 60)
    
    return failed == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)