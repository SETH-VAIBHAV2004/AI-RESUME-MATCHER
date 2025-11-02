import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class GapFinderNLP:
    """
    Novel GapFinder-NLP model based on fine-tuned BERT for semantic alignment
    between resumes and job descriptions.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        """
        Initialize GapFinder-NLP model.
        
        Args:
            model_name (str): Base BERT model name
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and base model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.bert_model = AutoModel.from_pretrained(model_name)
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Using fallback model...")
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.bert_model = AutoModel.from_pretrained('distilbert-base-uncased')
        
        # Initialize custom layers for gap analysis
        self.gap_classifier = self._build_gap_classifier()
        self.compatibility_scorer = self._build_compatibility_scorer()
        
        # Move to device
        self.bert_model.to(self.device)
        self.gap_classifier.to(self.device)
        self.compatibility_scorer.to(self.device)
        
        # Set to evaluation mode
        self.bert_model.eval()
        self.gap_classifier.eval()
        self.compatibility_scorer.eval()
    
    def _build_gap_classifier(self) -> nn.Module:
        """Build neural network for gap classification."""
        return nn.Sequential(
            nn.Linear(768, 512),  # BERT hidden size to 512
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 4),  # 4 gap categories: technical, tools, soft, other
            nn.Softmax(dim=1)
        )
    
    def _build_compatibility_scorer(self) -> nn.Module:
        """Build neural network for compatibility scoring."""
        return nn.Sequential(
            nn.Linear(1536, 512),  # Concatenated embeddings (768*2)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),  # Single compatibility score
            nn.Sigmoid()
        )
    
    def get_bert_embeddings(self, text: str) -> torch.Tensor:
        """
        Get BERT embeddings for input text.
        
        Args:
            text (str): Input text
            
        Returns:
            torch.Tensor: BERT embeddings
        """
        if not text.strip():
            return torch.zeros(768, device=self.device)
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            max_length=self.max_length,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings.squeeze()
    
    def predict_compatibility(self, resume_text: str, job_text: str) -> float:
        """
        Predict compatibility score between resume and job description.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            float: Compatibility probability (0-1)
        """
        # Get embeddings
        resume_embedding = self.get_bert_embeddings(resume_text)
        job_embedding = self.get_bert_embeddings(job_text)
        
        # Concatenate embeddings
        combined_embedding = torch.cat([resume_embedding, job_embedding], dim=0)
        combined_embedding = combined_embedding.unsqueeze(0)  # Add batch dimension
        
        # Predict compatibility
        with torch.no_grad():
            compatibility_score = self.compatibility_scorer(combined_embedding)
        
        return float(compatibility_score.item())
    
    def analyze_skill_gaps(self, resume_text: str, job_text: str) -> Dict[str, float]:
        """
        Analyze skill gaps using the gap classifier.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            Dict[str, float]: Gap probabilities by category
        """
        # Create gap context by highlighting differences
        gap_context = f"Resume: {resume_text[:500]} [SEP] Job Requirements: {job_text[:500]}"
        
        # Get embeddings for gap context
        gap_embedding = self.get_bert_embeddings(gap_context)
        gap_embedding = gap_embedding.unsqueeze(0)  # Add batch dimension
        
        # Predict gap categories
        with torch.no_grad():
            gap_probs = self.gap_classifier(gap_embedding)
        
        gap_categories = ['technical', 'tools', 'soft', 'other']
        gap_analysis = {
            category: float(prob) 
            for category, prob in zip(gap_categories, gap_probs.squeeze())
        }
        
        return gap_analysis
    
    def semantic_alignment_score(self, resume_text: str, job_text: str) -> Dict[str, float]:
        """
        Calculate semantic alignment scores using multiple metrics.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            Dict[str, float]: Alignment scores
        """
        # Get individual embeddings
        resume_embedding = self.get_bert_embeddings(resume_text).cpu().numpy()
        job_embedding = self.get_bert_embeddings(job_text).cpu().numpy()
        
        # Calculate cosine similarity
        cosine_sim = cosine_similarity(
            resume_embedding.reshape(1, -1),
            job_embedding.reshape(1, -1)
        )[0, 0]
        
        # Get compatibility prediction
        compatibility = self.predict_compatibility(resume_text, job_text)
        
        # Analyze skill gaps
        gap_analysis = self.analyze_skill_gaps(resume_text, job_text)
        
        # Calculate weighted alignment score
        alignment_score = (
            0.4 * cosine_sim +
            0.4 * compatibility +
            0.2 * (1 - max(gap_analysis.values()))  # Lower gap = higher alignment
        )
        
        return {
            'semantic_similarity': float(cosine_sim),
            'compatibility_score': compatibility,
            'alignment_score': float(alignment_score),
            'gap_analysis': gap_analysis
        }
    
    def generate_improvement_suggestions(self, gap_analysis: Dict[str, float], 
                                       threshold: float = 0.3) -> List[str]:
        """
        Generate improvement suggestions based on gap analysis.
        
        Args:
            gap_analysis (Dict[str, float]): Gap analysis results
            threshold (float): Threshold for suggesting improvements
            
        Returns:
            List[str]: List of improvement suggestions
        """
        suggestions = []
        
        if gap_analysis.get('technical', 0) > threshold:
            suggestions.extend([
                "Consider learning popular programming languages like Python, JavaScript, or Java",
                "Gain experience with modern frameworks and libraries",
                "Practice data structures and algorithms",
                "Build projects to demonstrate technical skills"
            ])
        
        if gap_analysis.get('tools', 0) > threshold:
            suggestions.extend([
                "Learn version control systems like Git",
                "Get familiar with cloud platforms (AWS, Azure, GCP)",
                "Practice with development tools and IDEs",
                "Explore CI/CD tools and DevOps practices"
            ])
        
        if gap_analysis.get('soft', 0) > threshold:
            suggestions.extend([
                "Develop leadership and communication skills",
                "Practice project management methodologies",
                "Improve teamwork and collaboration abilities",
                "Work on problem-solving and critical thinking"
            ])
        
        if gap_analysis.get('other', 0) > threshold:
            suggestions.extend([
                "Gain industry-specific knowledge and certifications",
                "Develop business acumen and domain expertise",
                "Improve analytical and research skills",
                "Consider additional training or education"
            ])
        
        return suggestions[:6]  # Return top 6 suggestions
    
    def comprehensive_analysis(self, resume_text: str, job_text: str) -> Dict:
        """
        Perform comprehensive analysis using GapFinder-NLP.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            Dict: Comprehensive analysis results
        """
        # Get semantic alignment scores
        alignment_results = self.semantic_alignment_score(resume_text, job_text)
        
        # Generate suggestions
        suggestions = self.generate_improvement_suggestions(
            alignment_results['gap_analysis']
        )
        
        # Calculate confidence score
        confidence = (
            alignment_results['semantic_similarity'] * 0.3 +
            alignment_results['compatibility_score'] * 0.4 +
            alignment_results['alignment_score'] * 0.3
        )
        
        return {
            'gapfinder_score': alignment_results['alignment_score'],
            'compatibility_probability': alignment_results['compatibility_score'],
            'semantic_similarity': alignment_results['semantic_similarity'],
            'skill_gap_analysis': alignment_results['gap_analysis'],
            'improvement_suggestions': suggestions,
            'confidence_score': float(confidence),
            'model_version': 'GapFinder-NLP v1.0'
        }