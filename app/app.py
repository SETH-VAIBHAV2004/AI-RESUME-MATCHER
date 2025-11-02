import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go
from io import BytesIO
import base64

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.text_cleaner import TextCleaner
from features.tfidf_features import TFIDFFeatures
from features.word2vec_features import Word2VecFeatures
from features.bert_features import BERTFeatures
from features.enhanced_gapfinder import EnhancedGapFinderNLP
from analysis.skill_extractor import SkillExtractor
from analysis.skill_gap import SkillGapAnalyzer
from evaluation.metrics import MatchingMetrics
from utils.file_parser import FileParser

# Page configuration
st.set_page_config(
    page_title="Resume-Job Description Matcher",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS for professional styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Header */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Metric Containers */
    .metric-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
    }
    
    /* Enhanced Skill Tags */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(79, 172, 254, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .skill-tag:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(79, 172, 254, 0.4);
    }
    
    .missing-skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        margin: 0.3rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: 0 2px 4px rgba(255, 107, 107, 0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .missing-skill-tag:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(255, 107, 107, 0.4);
    }
    
    /* Enhanced Recommendation Box */
    .recommendation-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin: 1rem 0;
        box-shadow: 0 8px 25px -5px rgba(102, 126, 234, 0.3);
        border-left: 4px solid #ffffff;
        position: relative;
        overflow: hidden;
    }
    
    .recommendation-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
        pointer-events: none;
    }
    
    /* Progress Bars */
    .progress-bar {
        background-color: #e2e8f0;
        border-radius: 10px;
        overflow: hidden;
        height: 8px;
        margin: 0.5rem 0;
    }
    
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 10px;
        transition: width 0.8s ease;
    }
    
    /* Enhanced Cards */
    .info-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px -10px rgba(0, 0, 0, 0.15);
    }
    
    /* Status Indicators */
    .status-success {
        color: #10b981;
        font-weight: 600;
    }
    
    .status-warning {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .status-error {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8fafc;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Enhanced Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* File Upload Enhancement */
    .stFileUploader > div > div {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border: 2px dashed #cbd5e1;
        border-radius: 16px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div > div:hover {
        border-color: #667eea;
        background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%);
    }
    
    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Success Animation */
    @keyframes slideInUp {
        from {
            transform: translateY(20px);
            opacity: 0;
        }
        to {
            transform: translateY(0);
            opacity: 1;
        }
    }
    
    .slide-in {
        animation: slideInUp 0.5s ease-out;
    }
    
    /* Enhanced Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%);
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    """Load and cache all models."""
    try:
        text_cleaner = TextCleaner()
        skill_extractor = SkillExtractor()
        skill_gap_analyzer = SkillGapAnalyzer()
        metrics_calculator = MatchingMetrics()
        
        return text_cleaner, skill_extractor, skill_gap_analyzer, metrics_calculator
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None, None

def create_pdf_report(results):
    """Create a PDF report of the analysis results."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        title = Paragraph("Resume-Job Description Match Report", styles['Title'])
        story.append(title)
        story.append(Spacer(1, 12))
        
        # Overall Score
        score_text = f"Overall Match Score: {results['final_score']:.1f}%"
        score_para = Paragraph(score_text, styles['Heading2'])
        story.append(score_para)
        story.append(Spacer(1, 12))
        
        # Model Scores
        model_scores = Paragraph("Model Scores:", styles['Heading3'])
        story.append(model_scores)
        
        for model, score in [
            ('TF-IDF', results['tfidf_score']),
            ('Word2Vec', results['word2vec_score']),
            ('BERT', results['bert_score']),
            ('GapFinder-NLP', results['gapfinder_score'])
        ]:
            model_text = f"• {model}: {score:.1f}%"
            story.append(Paragraph(model_text, styles['Normal']))
        
        story.append(Spacer(1, 12))
        
        # Recommendations
        if results.get('recommendations'):
            rec_title = Paragraph("Recommendations:", styles['Heading3'])
            story.append(rec_title)
            
            for i, rec in enumerate(results['recommendations'][:5], 1):
                rec_text = f"{i}. {rec['recommendation']}"
                story.append(Paragraph(rec_text, styles['Normal']))
        
        doc.build(story)
        buffer.seek(0)
        return buffer
    except ImportError:
        st.warning("PDF generation requires reportlab. Install with: pip install reportlab")
        return None
    except Exception as e:
        st.error(f"Error creating PDF: {str(e)}")
        return None

def analyze_resume_job_match(resume_text, job_text, text_cleaner, skill_extractor, 
                           skill_gap_analyzer, metrics_calculator):
    """Perform comprehensive resume-job matching analysis."""
    
    with st.spinner("🔄 Processing texts and extracting features..."):
        # Clean and preprocess texts
        resume_clean, resume_tokens = text_cleaner.preprocess_for_matching(resume_text)
        job_clean, job_tokens = text_cleaner.preprocess_for_matching(job_text)
        
        # Remove common resume words
        resume_tokens = text_cleaner.remove_common_resume_words(resume_tokens)
        job_tokens = text_cleaner.remove_common_resume_words(job_tokens)
    
    with st.spinner("🧠 Calculating similarity scores..."):
        # Initialize feature extractors
        tfidf_extractor = TFIDFFeatures()
        word2vec_extractor = Word2VecFeatures()
        bert_extractor = BERTFeatures()
        gapfinder_model = EnhancedGapFinderNLP()
        
        # Calculate similarity scores
        tfidf_score = tfidf_extractor.calculate_similarity(resume_clean, job_clean)
        word2vec_score = word2vec_extractor.calculate_similarity(resume_tokens, job_tokens)
        bert_score = bert_extractor.calculate_similarity(resume_clean, job_clean)
        
        # GapFinder-NLP analysis
        gapfinder_results = gapfinder_model.comprehensive_analysis(resume_clean, job_clean)
        gapfinder_score = gapfinder_results['gapfinder_score']
    
    with st.spinner("🔍 Extracting and analyzing skills..."):
        # Extract skills
        resume_skills = skill_extractor.extract_skills_hybrid(resume_clean)
        job_skills = skill_extractor.extract_skills_hybrid(job_clean)
        
        # Format skills for analysis
        resume_skills_formatted = skill_extractor.format_skills_for_display(resume_skills)
        job_skills_formatted = skill_extractor.format_skills_for_display(job_skills)
        
        # Analyze skill gaps
        gap_analysis = skill_gap_analyzer.analyze_skill_gaps(
            resume_skills_formatted, job_skills_formatted
        )
        
        # Generate recommendations
        recommendations = skill_gap_analyzer.generate_recommendations(gap_analysis)
    
    with st.spinner("📊 Calculating final metrics..."):
        # Calculate comprehensive metrics
        similarity_metrics = metrics_calculator.calculate_similarity_metrics(
            tfidf_score, word2vec_score, bert_score, gapfinder_score
        )
        
        # Generate performance report
        performance_report = metrics_calculator.generate_performance_report(
            similarity_metrics, gap_analysis
        )
    
    return {
        'similarity_metrics': similarity_metrics,
        'gap_analysis': gap_analysis,
        'recommendations': recommendations,
        'performance_report': performance_report,
        'resume_skills': resume_skills_formatted,
        'job_skills': job_skills_formatted,
        'gapfinder_results': gapfinder_results,
        'final_score': similarity_metrics['final_score'],
        'tfidf_score': similarity_metrics['tfidf_score'],
        'word2vec_score': similarity_metrics['word2vec_score'],
        'bert_score': similarity_metrics['bert_score'],
        'gapfinder_score': similarity_metrics['gapfinder_score']
    }

def main():
    """Main Streamlit application."""
    
    # Enhanced Header with subtitle
    st.markdown("""
    <div style="text-align: center; margin-bottom: 3rem;">
        <h1 class="main-header">🎯 Resume–Job Description Matcher</h1>
        <p style="font-size: 1.2rem; color: #64748b; font-weight: 400; margin-top: -1rem;">
            AI-Powered Skill Gap Analysis & Career Recommendations
        </p>
        <div style="width: 100px; height: 4px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); 
                    margin: 1rem auto; border-radius: 2px;"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    text_cleaner, skill_extractor, skill_gap_analyzer, metrics_calculator = load_models()
    
    if not all([text_cleaner, skill_extractor, skill_gap_analyzer, metrics_calculator]):
        st.error("Failed to load required models. Please check your installation.")
        return
    
    # Enhanced Sidebar
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 2rem;">
            <h2 style="color: white; margin: 0;">🚀 Quick Guide</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
            <h4 style="color: white; margin-bottom: 1rem;">📋 How to Use</h4>
            <div style="color: rgba(255,255,255,0.9); line-height: 1.6;">
                <p><strong>1.</strong> Upload or paste your resume</p>
                <p><strong>2.</strong> Upload or paste job description</p>
                <p><strong>3.</strong> Click "Analyze Match"</p>
                <p><strong>4.</strong> Review results & recommendations</p>
                <p><strong>5.</strong> Download your report</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px; margin-bottom: 2rem;">
            <h4 style="color: white; margin-bottom: 1rem;">🤖 AI Models</h4>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; line-height: 1.5;">
                <p><strong>🧠 BERT:</strong> Semantic understanding</p>
                <p><strong>📊 TF-IDF:</strong> Keyword matching</p>
                <p><strong>🔤 Word2Vec:</strong> Word relationships</p>
                <p><strong>⚡ GapFinder-NLP:</strong> Custom AI trained on 2,150+ real samples</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background: rgba(255,255,255,0.1); padding: 1.5rem; border-radius: 12px;">
            <h4 style="color: white; margin-bottom: 1rem;">📈 Features</h4>
            <div style="color: rgba(255,255,255,0.9); font-size: 0.9rem; line-height: 1.5;">
                <p>✅ Real-time analysis</p>
                <p>✅ PDF/DOCX support</p>
                <p>✅ Skill gap identification</p>
                <p>✅ Career recommendations</p>
                <p>✅ Downloadable reports</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload & Analyze", "📊 Results Dashboard", "💡 Recommendations"])
    
    with tab1:
        st.markdown("""
        <div class="info-card">
            <h2 style="margin-top: 0; color: #1e293b; display: flex; align-items: center;">
                📤 Upload & Analyze
                <span style="margin-left: auto; font-size: 0.8rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                      color: white; padding: 0.3rem 0.8rem; border-radius: 20px;">Step 1</span>
            </h2>
            <p style="color: #64748b; margin-bottom: 2rem;">Upload your documents or paste text directly for AI-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2, gap="large")
        
        with col1:
            st.markdown("""
            <div style="background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;
                        border: 2px solid #e2e8f0;">
                <h3 style="margin: 0; color: #1e293b; display: flex; align-items: center;">
                    📄 Your Resume
                    <span style="margin-left: auto; font-size: 0.7rem; background: #3b82f6; 
                          color: white; padding: 0.2rem 0.6rem; border-radius: 12px;">Required</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            resume_input_method = st.radio(
                "Choose input method:", 
                ["Paste Text", "Upload File"], 
                key="resume",
                help="Upload PDF/DOCX files or paste text directly"
            )
            
            if resume_input_method == "Paste Text":
                resume_text = st.text_area(
                    "Paste your resume text here:",
                    height=300,
                    placeholder="Enter your resume content..."
                )
            else:
                uploaded_resume = st.file_uploader(
                    "Upload resume file", 
                    type=['txt', 'pdf', 'docx'],
                    key="resume_file"
                )
                if uploaded_resume:
                    with st.spinner("📄 Extracting text from resume file..."):
                        resume_text = FileParser.parse_uploaded_file(uploaded_resume)
                        
                        if resume_text and FileParser.validate_extracted_text(resume_text):
                            st.success(f"✅ Successfully extracted {len(resume_text)} characters from {uploaded_resume.name}")
                            
                            # Show preview
                            with st.expander("📖 Preview extracted text"):
                                st.text(FileParser.preview_text(resume_text, 300))
                        elif resume_text:
                            st.warning("⚠️ Extracted text seems incomplete. Please verify the content or try pasting text directly.")
                            resume_text = resume_text or ""
                        else:
                            st.error("❌ Failed to extract text from file. Please try pasting text directly.")
                            resume_text = ""
                else:
                    resume_text = ""
        
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                        padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;
                        border: 2px solid #e2e8f0;">
                <h3 style="margin: 0; color: #1e293b; display: flex; align-items: center;">
                    💼 Job Description
                    <span style="margin-left: auto; font-size: 0.7rem; background: #3b82f6; 
                          color: white; padding: 0.2rem 0.6rem; border-radius: 12px;">Required</span>
                </h3>
            </div>
            """, unsafe_allow_html=True)
            
            job_input_method = st.radio(
                "Choose input method:", 
                ["Paste Text", "Upload File"], 
                key="job",
                help="Upload PDF/DOCX files or paste text directly"
            )
            
            if job_input_method == "Paste Text":
                job_text = st.text_area(
                    "Paste job description here:",
                    height=300,
                    placeholder="Enter job description content..."
                )
            else:
                uploaded_job = st.file_uploader(
                    "Upload job description file", 
                    type=['txt', 'pdf', 'docx'],
                    key="job_file"
                )
                if uploaded_job:
                    with st.spinner("📄 Extracting text from job description file..."):
                        job_text = FileParser.parse_uploaded_file(uploaded_job)
                        
                        if job_text and FileParser.validate_extracted_text(job_text):
                            st.success(f"✅ Successfully extracted {len(job_text)} characters from {uploaded_job.name}")
                            
                            # Show preview
                            with st.expander("📖 Preview extracted text"):
                                st.text(FileParser.preview_text(job_text, 300))
                        elif job_text:
                            st.warning("⚠️ Extracted text seems incomplete. Please verify the content or try pasting text directly.")
                            job_text = job_text or ""
                        else:
                            st.error("❌ Failed to extract text from file. Please try pasting text directly.")
                            job_text = ""
                else:
                    job_text = ""
        
        # Enhanced Analyze button
        st.markdown("<br>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            analyze_button = st.button(
                "🚀 Analyze Match", 
                type="primary", 
                use_container_width=True,
                help="Start AI-powered analysis of your resume against the job description"
            )
            
        if analyze_button and (not resume_text.strip() or not job_text.strip()):
            st.markdown("""
            <div style="background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%); 
                        color: #92400e; padding: 1rem; border-radius: 12px; margin: 1rem 0;
                        border-left: 4px solid #f59e0b;">
                <strong>⚠️ Missing Information</strong><br>
                Please provide both resume and job description texts to proceed with the analysis.
            </div>
            """, unsafe_allow_html=True)
        
        elif analyze_button:
            # Store results in session state with enhanced loading
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                status_text.text("🔄 Initializing AI models...")
                progress_bar.progress(20)
                
                status_text.text("🧠 Processing with BERT & GapFinder-NLP...")
                progress_bar.progress(50)
                
                results = analyze_resume_job_match(
                    resume_text, job_text, text_cleaner, skill_extractor,
                    skill_gap_analyzer, metrics_calculator
                )
                
                status_text.text("📊 Calculating final scores...")
                progress_bar.progress(80)
                
                st.session_state['analysis_results'] = results
                st.session_state['resume_text'] = resume_text
                st.session_state['job_text'] = job_text
                
                progress_bar.progress(100)
                status_text.empty()
                progress_bar.empty()
                
                st.markdown("""
                <div class="slide-in" style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                            color: white; padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
                            box-shadow: 0 8px 25px -5px rgba(16, 185, 129, 0.3);">
                    <h4 style="margin: 0; display: flex; align-items: center;">
                        ✅ Analysis Completed Successfully!
                        <span style="margin-left: auto; font-size: 0.8rem; background: rgba(255,255,255,0.2); 
                              padding: 0.3rem 0.8rem; border-radius: 20px;">Ready</span>
                    </h4>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                        Your comprehensive analysis is ready. Check the <strong>Results Dashboard</strong> tab for detailed insights.
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.markdown("""
                <div style="background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); 
                            color: white; padding: 1.5rem; border-radius: 16px; margin: 1rem 0;">
                    <h4 style="margin: 0;">❌ Analysis Error</h4>
                    <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">
                        An error occurred during analysis. Please try again or contact support.
                    </p>
                </div>
                """, unsafe_allow_html=True)
    
    with tab2:
        if 'analysis_results' not in st.session_state:
            st.markdown("""
            <div class="info-card" style="text-align: center; padding: 3rem;">
                <h2 style="color: #64748b; margin-bottom: 1rem;">📊 Results Dashboard</h2>
                <div style="font-size: 4rem; margin: 2rem 0; opacity: 0.3;">📈</div>
                <h3 style="color: #94a3b8;">No Analysis Yet</h3>
                <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
                    Upload your resume and job description in the <strong>Upload & Analyze</strong> tab to see detailed results here.
                </p>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 1rem; border-radius: 12px; display: inline-block;">
                    👆 Start by uploading your documents
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            results = st.session_state['analysis_results']
            
            # Enhanced Header for Results
            st.markdown("""
            <div class="info-card">
                <h2 style="margin-top: 0; color: #1e293b; display: flex; align-items: center;">
                    📊 Analysis Results
                    <span style="margin-left: auto; font-size: 0.8rem; background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                          color: white; padding: 0.3rem 0.8rem; border-radius: 20px;">Complete</span>
                </h2>
                <p style="color: #64748b; margin-bottom: 0;">Comprehensive AI-powered analysis of your resume-job match</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Overall Score Section
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                final_score = results['final_score']
                
                # Determine score category and colors
                if final_score >= 85:
                    score_category = "Excellent Match"
                    score_color = "#10b981"
                    bg_color = "#d1fae5"
                    icon = "🎉"
                elif final_score >= 70:
                    score_category = "Good Match"
                    score_color = "#3b82f6"
                    bg_color = "#dbeafe"
                    icon = "👍"
                elif final_score >= 55:
                    score_category = "Fair Match"
                    score_color = "#f59e0b"
                    bg_color = "#fef3c7"
                    icon = "⚡"
                else:
                    score_category = "Needs Improvement"
                    score_color = "#ef4444"
                    bg_color = "#fee2e2"
                    icon = "🚀"
                
                # Simplified score display to avoid HTML rendering issues
                interpretation = ("🎯 You're an excellent fit for this role!" if final_score >= 85 else
                                "✅ You meet most requirements with minor gaps." if final_score >= 70 else
                                "⚡ You have potential but need skill development." if final_score >= 55 else
                                "🚀 Significant upskilling needed for this role.")
                
                st.markdown(f"""
                <div style="background: {bg_color}; padding: 2rem; border-radius: 20px; 
                            border: 3px solid {score_color}; text-align: center; margin-bottom: 1rem;">
                    <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
                    <div style="font-size: 4rem; font-weight: bold; color: {score_color}; margin-bottom: 0.5rem;">
                        {final_score:.0f}%
                    </div>
                    <div style="font-size: 1.3rem; font-weight: 600; color: {score_color}; margin-bottom: 1rem;">
                        {score_category}
                    </div>
                    <div style="background: rgba(255,255,255,0.7); border-radius: 25px; padding: 0.3rem; margin-bottom: 1rem;">
                        <div style="background: {score_color}; height: 12px; border-radius: 20px; 
                                    width: {final_score}%; transition: width 1s ease;"></div>
                    </div>
                    <div style="background: rgba(255,255,255,0.8); padding: 1rem; border-radius: 12px; 
                                font-size: 0.9rem; color: #374151;">
                        <strong>What this means:</strong><br>
                        {interpretation}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                grade = results['performance_report']['overall_grade']
                level = results['performance_report']['performance_level']
                
                # Enhanced grade display
                grade_color = "#10b981" if grade in ['A+', 'A'] else "#f59e0b" if grade == 'B' else "#ef4444"
                
                st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <h3 style="margin-top: 0; color: #1e293b;">📈 Grade</h3>
                    <div style="font-size: 3rem; font-weight: bold; color: {grade_color}; margin: 1rem 0;">
                        {grade}
                    </div>
                    <p style="color: #64748b; margin: 0; font-weight: 500;">{level}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                matched = results['performance_report']['total_skills_matched']
                missing = results['performance_report']['total_skills_missing']
                
                st.markdown(f"""
                <div class="metric-container" style="text-align: center;">
                    <h3 style="margin-top: 0; color: #1e293b;">🎯 Skills</h3>
                    <div style="font-size: 2.5rem; font-weight: bold; color: #3b82f6; margin: 1rem 0;">
                        {matched}
                    </div>
                    <p style="color: #64748b; margin: 0;">
                        <span style="color: #10b981; font-weight: 600;">✓ {matched} matched</span><br>
                        <span style="color: #ef4444; font-weight: 600;">✗ {missing} missing</span>
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # Enhanced Score Breakdown
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #1e293b;">📊 Score Breakdown</h3>
                <p style="color: #64748b; margin-bottom: 1.5rem;">How your final score was calculated using different AI models</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Individual model scores with explanations
            model_scores = [
                ("🧠 BERT Semantic Analysis", results['similarity_metrics']['bert_score'], 
                 "Understands the meaning and context of your experience"),
                ("📊 TF-IDF Keyword Matching", results['similarity_metrics']['tfidf_score'], 
                 "Matches specific keywords and technical terms"),
                ("🔤 Word2Vec Relationships", results['similarity_metrics']['word2vec_score'], 
                 "Finds related skills and synonymous terms"),
                ("⚡ GapFinder-NLP AI", results['similarity_metrics']['gapfinder_score'], 
                 "Custom AI trained on 2,150+ real job matches")
            ]
            
            for model_name, score, description in model_scores:
                score_width = score
                score_color = "#10b981" if score >= 80 else "#3b82f6" if score >= 60 else "#f59e0b" if score >= 40 else "#ef4444"
                
                st.markdown(f"""
                <div style="background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                            padding: 1.5rem; border-radius: 16px; margin-bottom: 1rem;
                            border-left: 4px solid {score_color};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.8rem;">
                        <div>
                            <strong style="color: #1e293b; font-size: 1.1rem;">{model_name}</strong>
                            <div style="color: #64748b; font-size: 0.9rem; margin-top: 0.2rem;">{description}</div>
                        </div>
                        <div style="font-size: 1.5rem; font-weight: bold; color: {score_color};">
                            {score:.1f}%
                        </div>
                    </div>
                    <div style="background: #e2e8f0; border-radius: 10px; overflow: hidden; height: 8px;">
                        <div style="background: {score_color}; height: 100%; width: {score_width}%; 
                                    transition: width 1s ease; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            # Final calculation explanation
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        color: white; padding: 1.5rem; border-radius: 16px; margin-top: 1.5rem;">
                <h4 style="margin: 0 0 1rem 0;">🧮 Final Score Calculation</h4>
                <div style="background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 12px;">
                    <strong>Weighted Average:</strong> TF-IDF (25%) + Word2Vec (25%) + BERT (25%) + GapFinder-NLP (25%)<br>
                    <strong>Your Result:</strong> {results['final_score']:.1f}% overall compatibility
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Interactive Skill Matching
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #1e293b;">🎯 Skill Matching Overview</h3>
                <p style="color: #64748b; margin-bottom: 1.5rem;">Interactive breakdown of your skills vs job requirements</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Create skill matching cards
            categories = ['technical', 'tools', 'soft', 'other']
            category_info = {
                'technical': {'icon': '🧠', 'name': 'Technical Skills', 'color': '#3b82f6'},
                'tools': {'icon': '🛠️', 'name': 'Tools & Platforms', 'color': '#10b981'},
                'soft': {'icon': '💬', 'name': 'Soft Skills', 'color': '#f59e0b'},
                'other': {'icon': '📄', 'name': 'Other Skills', 'color': '#8b5cf6'}
            }
            
            cols = st.columns(4)
            
            for i, category in enumerate(categories):
                gap_data = results['gap_analysis'][category]
                info = category_info[category]
                
                with cols[i]:
                    match_rate = gap_data['match_percentage']
                    matched = gap_data['total_matched']
                    required = gap_data['total_required']
                    
                    # Determine status
                    if match_rate >= 80:
                        status = "Excellent"
                        status_color = "#10b981"
                    elif match_rate >= 60:
                        status = "Good"
                        status_color = "#3b82f6"
                    elif match_rate >= 40:
                        status = "Fair"
                        status_color = "#f59e0b"
                    else:
                        status = "Poor"
                        status_color = "#ef4444"
                    
                    # Use Streamlit's built-in metric instead of complex HTML
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                                padding: 1.5rem; border-radius: 16px; text-align: center;
                                border: 2px solid {info['color']}; margin-bottom: 1rem;">
                        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{info['icon']}</div>
                        <h4 style="color: #1e293b; margin: 0.5rem 0; font-size: 1rem;">{info['name']}</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Simple progress bar instead of SVG
                    st.markdown(f"""
                    <div style="text-align: center; padding: 0 1rem;">
                        <div style="font-size: 2rem; font-weight: bold; color: {status_color}; margin: 0.5rem 0;">
                            {match_rate:.0f}%
                        </div>
                        <div style="background: #e2e8f0; border-radius: 10px; overflow: hidden; height: 8px; margin: 0.5rem 0;">
                            <div style="background: {status_color}; height: 100%; width: {match_rate}%; 
                                        transition: width 1s ease; border-radius: 10px;"></div>
                        </div>
                        <div style="color: {status_color}; font-weight: 600; margin-bottom: 0.5rem;">{status}</div>
                        <div style="color: #64748b; font-size: 0.9rem;">{matched}/{required} skills</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Enhanced Detailed Skill Breakdown
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #1e293b;">🔍 Detailed Skill Analysis</h3>
                <p style="color: #64748b; margin-bottom: 1rem;">Comprehensive breakdown of skills by category with match rates</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Category icons
            category_icons = {
                'technical': '🧠',
                'tools': '🛠️', 
                'soft': '💬',
                'other': '📄'
            }
            
            for category in ['technical', 'tools', 'soft', 'other']:
                gap_data = results['gap_analysis'][category]
                match_rate = gap_data['match_percentage']
                
                # Color coding based on match rate
                if match_rate >= 80:
                    color = "#10b981"
                    status = "Excellent"
                elif match_rate >= 60:
                    color = "#f59e0b" 
                    status = "Good"
                else:
                    color = "#ef4444"
                    status = "Needs Improvement"
                
                with st.expander(f"{category_icons[category]} {category.title()} Skills - {match_rate:.1f}% Match ({status})"):
                    # Progress bar for match rate
                    st.markdown(f"""
                    <div style="margin-bottom: 1.5rem;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 0.5rem;">
                            <span style="font-weight: 600; color: #1e293b;">Match Rate</span>
                            <span style="font-weight: 600; color: {color};">{match_rate:.1f}%</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {match_rate}%; background: {color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**✅ Matched Skills:**")
                        if gap_data['matched_skills']:
                            skills_html = ""
                            for skill in gap_data['matched_skills']:
                                skills_html += f'<span class="skill-tag">✓ {skill.title()}</span>'
                            st.markdown(skills_html, unsafe_allow_html=True)
                        else:
                            st.markdown("*No matched skills found in this category.*")
                    
                    with col2:
                        st.markdown("**❌ Missing Skills:**")
                        if gap_data['missing_skills']:
                            skills_html = ""
                            for skill in gap_data['missing_skills']:
                                skills_html += f'<span class="missing-skill-tag">✗ {skill.title()}</span>'
                            st.markdown(skills_html, unsafe_allow_html=True)
                        else:
                            st.markdown("*Perfect match! No missing skills.*")
                    
                    # Enhanced metrics display
                    st.markdown(f"""
                    <div style="background: linear-gradient(145deg, #f8fafc 0%, #f1f5f9 100%); 
                                padding: 1rem; border-radius: 12px; margin-top: 1rem;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <strong style="color: #1e293b;">Category Summary</strong><br>
                                <span style="color: #64748b;">
                                    {gap_data['total_matched']} of {gap_data['total_required']} required skills matched
                                </span>
                            </div>
                            <div style="text-align: right;">
                                <div style="font-size: 1.5rem; font-weight: bold; color: {color};">
                                    {gap_data['total_matched']}/{gap_data['total_required']}
                                </div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with tab3:
        if 'analysis_results' not in st.session_state:
            st.markdown("""
            <div class="info-card" style="text-align: center; padding: 3rem;">
                <h2 style="color: #64748b; margin-bottom: 1rem;">💡 Recommendations</h2>
                <div style="font-size: 4rem; margin: 2rem 0; opacity: 0.3;">🎯</div>
                <h3 style="color: #94a3b8;">No Recommendations Yet</h3>
                <p style="color: #64748b; font-size: 1.1rem; margin-bottom: 2rem;">
                    Complete your analysis first to receive personalized career recommendations and improvement suggestions.
                </p>
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            color: white; padding: 1rem; border-radius: 12px; display: inline-block;">
                    🚀 Start your analysis to unlock insights
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            results = st.session_state['analysis_results']
            
            # Summary Insight
            st.subheader("📝 Summary Insight")
            performance_level = results['performance_report']['performance_level']
            final_score = results['final_score']
            
            if final_score >= 80:
                insight_color = "success"
                insight_icon = "🎉"
            elif final_score >= 60:
                insight_color = "warning"
                insight_icon = "⚡"
            else:
                insight_color = "error"
                insight_icon = "🚀"
            
            st.markdown(f"""
            <div class="recommendation-box">
                <h4>{insight_icon} Overall Assessment</h4>
                <p>Your resume shows a <strong>{performance_level.lower()}</strong> match with the job description, 
                scoring <strong>{final_score:.1f}%</strong>. The GapFinder-NLP model identified key areas for improvement 
                to enhance your candidacy.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Enhanced Personalized Recommendations
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #1e293b;">🎯 Personalized Action Plan</h3>
                <p style="color: #64748b; margin-bottom: 1.5rem;">Prioritized recommendations to improve your job match score</p>
            </div>
            """, unsafe_allow_html=True)
            
            recommendations = results['recommendations']
            if recommendations:
                # Group recommendations by priority
                high_priority = [r for r in recommendations if r.get('priority', 0) >= 3]
                medium_priority = [r for r in recommendations if r.get('priority', 0) == 2]
                low_priority = [r for r in recommendations if r.get('priority', 0) <= 1]
                
                priority_groups = [
                    ("🔥 High Priority", high_priority, "#ef4444", "Focus on these first for maximum impact"),
                    ("⚡ Medium Priority", medium_priority, "#f59e0b", "Important for strengthening your profile"),
                    ("💡 Low Priority", low_priority, "#3b82f6", "Nice to have for competitive advantage")
                ]
                
                for group_name, group_recs, color, description in priority_groups:
                    if group_recs:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, {color} 0%, {color}dd 100%); 
                                    color: white; padding: 1rem; border-radius: 12px; margin: 1rem 0;">
                            <h4 style="margin: 0; display: flex; align-items: center;">
                                {group_name}
                                <span style="margin-left: auto; font-size: 0.8rem; background: rgba(255,255,255,0.2); 
                                      padding: 0.3rem 0.8rem; border-radius: 20px;">{len(group_recs)} items</span>
                            </h4>
                            <p style="margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;">{description}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        for i, rec in enumerate(group_recs, 1):
                            category_icons = {
                                'technical': '🧠',
                                'tools': '🛠️',
                                'soft': '💬',
                                'other': '📄'
                            }
                            
                            icon = category_icons.get(rec['category'], '💡')
                            
                            st.markdown(f"""
                            <div style="background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                                        padding: 1.5rem; border-radius: 16px; margin: 1rem 0;
                                        border-left: 4px solid {color};">
                                <div style="display: flex; align-items: start; gap: 1rem;">
                                    <div style="font-size: 2rem; flex-shrink: 0;">{icon}</div>
                                    <div style="flex-grow: 1;">
                                        <h4 style="margin: 0 0 0.5rem 0; color: #1e293b;">
                                            {rec['skill'].title()}
                                            <span style="font-size: 0.7rem; background: {color}; color: white; 
                                                  padding: 0.2rem 0.6rem; border-radius: 12px; margin-left: 0.5rem;">
                                                {rec['category'].title()}
                                            </span>
                                        </h4>
                                        <p style="color: #64748b; margin: 0.5rem 0; line-height: 1.5;">
                                            {rec['recommendation']}
                                        </p>
                                        <div style="display: flex; gap: 1rem; margin-top: 1rem;">
                                            <div style="background: #f1f5f9; padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.9rem;">
                                                <strong>Priority:</strong> {rec.get('priority', 'N/A')}/4
                                            </div>
                                            <div style="background: #f1f5f9; padding: 0.5rem 1rem; border-radius: 8px; font-size: 0.9rem;">
                                                <strong>Impact:</strong> {"High" if rec.get('priority', 0) >= 3 else "Medium" if rec.get('priority', 0) == 2 else "Low"}
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #10b981 0%, #059669 100%); 
                            color: white; padding: 2rem; border-radius: 16px; text-align: center;">
                    <div style="font-size: 3rem; margin-bottom: 1rem;">🎉</div>
                    <h3 style="margin: 0 0 1rem 0;">Excellent Match!</h3>
                    <p style="margin: 0; opacity: 0.9;">
                        No specific skill gaps identified. You're well-qualified for this position!
                    </p>
                </div>
                """, unsafe_allow_html=True)
            
            # GapFinder-NLP Insights
            if 'gapfinder_results' in results:
                st.subheader("🤖 GapFinder-NLP Insights")
                gf_results = results['gapfinder_results']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Compatibility Score", f"{gf_results['compatibility_probability']:.1f}%")
                    st.metric("Semantic Similarity", f"{gf_results['semantic_similarity']:.1f}%")
                
                with col2:
                    st.metric("Confidence Score", f"{gf_results['confidence_score']:.1f}%")
                    st.metric("Model Version", gf_results['model_version'])
                
                # Improvement suggestions from GapFinder
                if gf_results.get('improvement_suggestions'):
                    st.write("**🚀 AI-Generated Improvement Suggestions:**")
                    for suggestion in gf_results['improvement_suggestions'][:4]:
                        st.write(f"• {suggestion}")
            
            # Download Report Section
            st.markdown("""
            <div class="info-card">
                <h3 style="margin-top: 0; color: #1e293b;">📥 Export Your Results</h3>
                <p style="color: #64748b; margin-bottom: 1.5rem;">Download your analysis in different formats</p>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                            padding: 1.5rem; border-radius: 16px; text-align: center;
                            border: 2px solid #3b82f6;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📊</div>
                    <h4 style="color: #1e293b; margin-bottom: 0.5rem;">JSON Data Report</h4>
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem;">
                        Machine-readable format for developers and data analysis
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # JSON Report
                json_report = {
                    'analysis_summary': {
                        'final_score': results['final_score'],
                        'performance_grade': results['performance_report']['overall_grade'],
                        'total_skills_matched': results['performance_report']['total_skills_matched'],
                        'total_skills_missing': results['performance_report']['total_skills_missing']
                    },
                    'model_scores': {
                        'tfidf': results['tfidf_score'],
                        'word2vec': results['word2vec_score'],
                        'bert': results['bert_score'],
                        'gapfinder': results['gapfinder_score']
                    },
                    'skill_gaps': results['gap_analysis'],
                    'recommendations': results['recommendations']
                }
                
                json_str = pd.Series(json_report).to_json(indent=2)
                st.download_button(
                    label="📄 Download JSON Report",
                    data=json_str,
                    file_name="resume_match_report.json",
                    mime="application/json",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("""
                <div style="background: linear-gradient(145deg, #ffffff 0%, #f8fafc 100%); 
                            padding: 1.5rem; border-radius: 16px; text-align: center;
                            border: 2px solid #ef4444;">
                    <div style="font-size: 2rem; margin-bottom: 1rem;">📑</div>
                    <h4 style="color: #1e293b; margin-bottom: 0.5rem;">PDF Summary Report</h4>
                    <p style="color: #64748b; font-size: 0.9rem; margin-bottom: 1.5rem;">
                        Professional report for sharing and printing
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # PDF Report
                pdf_buffer = create_pdf_report(results)
                if pdf_buffer:
                    st.download_button(
                        label="📑 Download PDF Report",
                        data=pdf_buffer,
                        file_name="resume_match_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                else:
                    st.button(
                        "📑 PDF Report (Setup Required)",
                        disabled=True,
                        use_container_width=True,
                        help="Install reportlab: pip install reportlab"
                    )

if __name__ == "__main__":
    main()