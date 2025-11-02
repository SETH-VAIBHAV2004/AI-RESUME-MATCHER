#!/usr/bin/env python3
"""
Main entry point for the Resume-Job Description Matcher application.
This script provides a command-line interface and can also launch the Streamlit app.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from preprocessing.text_cleaner import TextCleaner
from features.tfidf_features import TFIDFFeatures
from features.word2vec_features import Word2VecFeatures
from features.bert_features import BERTFeatures
from features.gapfinder_nlp import GapFinderNLP
from analysis.skill_extractor import SkillExtractor
from analysis.skill_gap import SkillGapAnalyzer
from evaluation.metrics import MatchingMetrics

def run_cli_analysis(resume_file: str, job_file: str):
    """
    Run analysis from command line with file inputs.
    
    Args:
        resume_file (str): Path to resume text file
        job_file (str): Path to job description text file
    """
    print("🚀 Starting Resume-Job Description Analysis...")
    
    # Read input files
    try:
        with open(resume_file, 'r', encoding='utf-8') as f:
            resume_text = f.read()
        with open(job_file, 'r', encoding='utf-8') as f:
            job_text = f.read()
    except FileNotFoundError as e:
        print(f"❌ Error: File not found - {e}")
        return
    except Exception as e:
        print(f"❌ Error reading files: {e}")
        return
    
    # Initialize components
    print("🔧 Initializing models...")
    text_cleaner = TextCleaner()
    skill_extractor = SkillExtractor()
    skill_gap_analyzer = SkillGapAnalyzer()
    metrics_calculator = MatchingMetrics()
    
    # Preprocess texts
    print("🔄 Preprocessing texts...")
    resume_clean, resume_tokens = text_cleaner.preprocess_for_matching(resume_text)
    job_clean, job_tokens = text_cleaner.preprocess_for_matching(job_text)
    
    # Remove common words
    resume_tokens = text_cleaner.remove_common_resume_words(resume_tokens)
    job_tokens = text_cleaner.remove_common_resume_words(job_tokens)
    
    # Calculate similarity scores
    print("🧠 Calculating similarity scores...")
    
    # TF-IDF
    tfidf_extractor = TFIDFFeatures()
    tfidf_score = tfidf_extractor.calculate_similarity(resume_clean, job_clean)
    
    # Word2Vec
    word2vec_extractor = Word2VecFeatures()
    word2vec_score = word2vec_extractor.calculate_similarity(resume_tokens, job_tokens)
    
    # BERT
    bert_extractor = BERTFeatures()
    bert_score = bert_extractor.calculate_similarity(resume_clean, job_clean)
    
    # GapFinder-NLP
    gapfinder_model = GapFinderNLP()
    gapfinder_results = gapfinder_model.comprehensive_analysis(resume_clean, job_clean)
    gapfinder_score = gapfinder_results['gapfinder_score']
    
    # Extract skills
    print("🔍 Extracting skills...")
    resume_skills = skill_extractor.extract_skills_hybrid(resume_clean)
    job_skills = skill_extractor.extract_skills_hybrid(job_clean)
    
    # Format skills
    resume_skills_formatted = skill_extractor.format_skills_for_display(resume_skills)
    job_skills_formatted = skill_extractor.format_skills_for_display(job_skills)
    
    # Analyze gaps
    print("📊 Analyzing skill gaps...")
    gap_analysis = skill_gap_analyzer.analyze_skill_gaps(
        resume_skills_formatted, job_skills_formatted
    )
    
    # Generate recommendations
    recommendations = skill_gap_analyzer.generate_recommendations(gap_analysis)
    
    # Calculate metrics
    similarity_metrics = metrics_calculator.calculate_similarity_metrics(
        tfidf_score, word2vec_score, bert_score, gapfinder_score
    )
    
    # Display results
    print("\n" + "="*60)
    print("📊 ANALYSIS RESULTS")
    print("="*60)
    
    print(f"\n🎯 OVERALL MATCH SCORE: {similarity_metrics['final_score']:.1f}%")
    print(f"📈 Performance Level: {get_performance_level(similarity_metrics['final_score'])}")
    
    print(f"\n🤖 MODEL SCORES:")
    print(f"   • TF-IDF:        {similarity_metrics['tfidf_score']:.1f}%")
    print(f"   • Word2Vec:      {similarity_metrics['word2vec_score']:.1f}%")
    print(f"   • BERT:          {similarity_metrics['bert_score']:.1f}%")
    print(f"   • GapFinder-NLP: {similarity_metrics['gapfinder_score']:.1f}%")
    
    print(f"\n📋 SKILL SUMMARY:")
    total_matched = sum(len(skills) for skills in resume_skills_formatted.values())
    total_required = sum(len(skills) for skills in job_skills_formatted.values())
    total_missing = sum(gap_analysis[cat]['total_missing'] for cat in gap_analysis)
    
    print(f"   • Skills Matched: {total_matched}")
    print(f"   • Skills Required: {total_required}")
    print(f"   • Skills Missing: {total_missing}")
    
    print(f"\n🔍 SKILL GAPS BY CATEGORY:")
    for category, analysis in gap_analysis.items():
        print(f"   • {category.title()}: {analysis['match_percentage']:.1f}% match "
              f"({analysis['total_matched']}/{analysis['total_required']} skills)")
    
    print(f"\n💡 TOP RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations[:5], 1):
        print(f"   {i}. {rec['recommendation']}")
    
    print(f"\n🤖 GAPFINDER-NLP INSIGHTS:")
    print(f"   • Compatibility: {gapfinder_results['compatibility_probability']:.1f}%")
    print(f"   • Semantic Similarity: {gapfinder_results['semantic_similarity']:.1f}%")
    print(f"   • Confidence: {gapfinder_results['confidence_score']:.1f}%")
    
    print("\n" + "="*60)
    print("✅ Analysis completed!")
    print("💡 For detailed interactive analysis, run: streamlit run app/app.py")
    print("="*60)

def get_performance_level(score: float) -> str:
    """Get performance level based on score."""
    if score >= 90:
        return "Excellent (A+)"
    elif score >= 80:
        return "Very Good (A)"
    elif score >= 70:
        return "Good (B)"
    elif score >= 60:
        return "Fair (C)"
    else:
        return "Needs Improvement (D)"

def launch_streamlit_app():
    """Launch the Streamlit web application."""
    import subprocess
    import sys
    
    app_path = os.path.join(os.path.dirname(__file__), 'app', 'app.py')
    
    print("🚀 Launching Streamlit application...")
    print("📱 The app will open in your default web browser.")
    print("🔗 If it doesn't open automatically, go to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the application.")
    
    try:
        subprocess.run([sys.executable, '-m', 'streamlit', 'run', app_path])
    except KeyboardInterrupt:
        print("\n👋 Application stopped.")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")
        print("💡 Make sure Streamlit is installed: pip install streamlit")

def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Resume-Job Description Matcher with Skill Gap Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch interactive web app
  python main.py --web
  
  # Analyze files from command line
  python main.py --resume resume.txt --job job_description.txt
  
  # Get help
  python main.py --help
        """
    )
    
    parser.add_argument(
        '--web', 
        action='store_true',
        help='Launch the interactive Streamlit web application'
    )
    
    parser.add_argument(
        '--resume', 
        type=str,
        help='Path to resume text file'
    )
    
    parser.add_argument(
        '--job', 
        type=str,
        help='Path to job description text file'
    )
    
    parser.add_argument(
        '--version', 
        action='version',
        version='Resume Matcher v1.0 - Powered by GapFinder-NLP'
    )
    
    args = parser.parse_args()
    
    # Show help if no arguments provided
    if len(sys.argv) == 1:
        parser.print_help()
        print("\n💡 Quick start: python main.py --web")
        return
    
    # Launch web app
    if args.web:
        launch_streamlit_app()
        return
    
    # CLI analysis
    if args.resume and args.job:
        run_cli_analysis(args.resume, args.job)
        return
    
    # Invalid arguments
    if args.resume or args.job:
        print("❌ Error: Both --resume and --job arguments are required for CLI analysis.")
        print("💡 Use --web for interactive analysis or provide both file paths.")
        parser.print_help()
        return
    
    parser.print_help()

if __name__ == "__main__":
    main()