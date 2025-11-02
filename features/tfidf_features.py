import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple, List

class TFIDFFeatures:
    def __init__(self, max_features: int = 5000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF feature extractor.
        
        Args:
            max_features (int): Maximum number of features
            ngram_range (Tuple[int, int]): N-gram range for feature extraction
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words='english',
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z0-9]*\b'
        )
        self.is_fitted = False
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the vectorizer and transform texts to TF-IDF vectors.
        
        Args:
            texts (List[str]): List of texts to transform
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        self.is_fitted = True
        return tfidf_matrix.toarray()
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to TF-IDF vectors using fitted vectorizer.
        
        Args:
            texts (List[str]): List of texts to transform
            
        Returns:
            np.ndarray: TF-IDF matrix
        """
        if not self.is_fitted:
            raise ValueError("Vectorizer must be fitted before transform")
        
        tfidf_matrix = self.vectorizer.transform(texts)
        return tfidf_matrix.toarray()
    
    def calculate_similarity(self, resume_text: str, job_text: str) -> float:
        """
        Calculate TF-IDF cosine similarity between resume and job description.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Combine texts for fitting
        texts = [resume_text, job_text]
        
        # Transform to TF-IDF vectors
        tfidf_matrix = self.fit_transform(texts)
        
        # Calculate cosine similarity
        similarity_matrix = cosine_similarity(tfidf_matrix)
        similarity_score = similarity_matrix[0, 1]
        
        return float(similarity_score)
    
    def get_top_features(self, text: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Get top TF-IDF features for a given text.
        
        Args:
            text (str): Input text
            top_n (int): Number of top features to return
            
        Returns:
            List[Tuple[str, float]]: List of (feature, score) tuples
        """
        if not self.is_fitted:
            # Fit on the single text
            self.fit_transform([text])
        
        # Transform the text
        tfidf_vector = self.transform([text])[0]
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get top features
        top_indices = np.argsort(tfidf_vector)[::-1][:top_n]
        top_features = [
            (feature_names[idx], tfidf_vector[idx]) 
            for idx in top_indices 
            if tfidf_vector[idx] > 0
        ]
        
        return top_features
    
    def get_feature_importance(self, resume_text: str, job_text: str) -> dict:
        """
        Get feature importance analysis for resume-job matching.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            dict: Feature importance analysis
        """
        # Calculate similarity
        similarity_score = self.calculate_similarity(resume_text, job_text)
        
        # Get top features for both texts
        resume_features = self.get_top_features(resume_text, 15)
        job_features = self.get_top_features(job_text, 15)
        
        # Find common features
        resume_terms = {term for term, _ in resume_features}
        job_terms = {term for term, _ in job_features}
        common_terms = resume_terms.intersection(job_terms)
        
        return {
            'similarity_score': similarity_score,
            'resume_top_terms': resume_features,
            'job_top_terms': job_features,
            'common_terms': list(common_terms),
            'resume_unique_terms': list(resume_terms - job_terms),
            'job_unique_terms': list(job_terms - resume_terms)
        }