import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class BERTFeatures:
    def __init__(self, model_name: str = 'bert-base-uncased', max_length: int = 512):
        """
        Initialize BERT feature extractor.
        
        Args:
            model_name (str): Pre-trained BERT model name
            max_length (int): Maximum sequence length
        """
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load tokenizer and model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            print(f"Error loading BERT model: {e}")
            print("Using fallback model...")
            self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
            self.model = AutoModel.from_pretrained('distilbert-base-uncased')
            self.model.to(self.device)
            self.model.eval()
    
    def get_embeddings(self, text: str) -> np.ndarray:
        """
        Get BERT embeddings for text.
        
        Args:
            text (str): Input text
            
        Returns:
            np.ndarray: BERT embeddings
        """
        if not text.strip():
            return np.zeros(768)  # BERT base hidden size
        
        # Tokenize text
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
            outputs = self.model(**inputs)
            # Use [CLS] token embedding as document representation
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embeddings.flatten()
    
    def calculate_similarity(self, resume_text: str, job_text: str) -> float:
        """
        Calculate BERT cosine similarity between resume and job description.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Get embeddings for both texts
        resume_embeddings = self.get_embeddings(resume_text)
        job_embeddings = self.get_embeddings(job_text)
        
        # Calculate cosine similarity
        similarity = cosine_similarity(
            resume_embeddings.reshape(1, -1),
            job_embeddings.reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def get_sentence_embeddings(self, sentences: List[str]) -> np.ndarray:
        """
        Get BERT embeddings for multiple sentences.
        
        Args:
            sentences (List[str]): List of sentences
            
        Returns:
            np.ndarray: Matrix of sentence embeddings
        """
        if not sentences:
            return np.array([])
        
        embeddings = []
        for sentence in sentences:
            embedding = self.get_embeddings(sentence)
            embeddings.append(embedding)
        
        return np.array(embeddings)
    
    def find_most_similar_sentences(self, resume_sentences: List[str], 
                                  job_sentences: List[str], 
                                  top_n: int = 5) -> List[Tuple[str, str, float]]:
        """
        Find most similar sentence pairs between resume and job description.
        
        Args:
            resume_sentences (List[str]): Resume sentences
            job_sentences (List[str]): Job description sentences
            top_n (int): Number of top similar pairs to return
            
        Returns:
            List[Tuple[str, str, float]]: List of (resume_sent, job_sent, similarity) tuples
        """
        if not resume_sentences or not job_sentences:
            return []
        
        # Get embeddings
        resume_embeddings = self.get_sentence_embeddings(resume_sentences)
        job_embeddings = self.get_sentence_embeddings(job_sentences)
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(resume_embeddings, job_embeddings)
        
        # Find top similar pairs
        similar_pairs = []
        for i, resume_sent in enumerate(resume_sentences):
            for j, job_sent in enumerate(job_sentences):
                similarity = similarity_matrix[i, j]
                similar_pairs.append((resume_sent, job_sent, similarity))
        
        # Sort by similarity and return top N
        similar_pairs.sort(key=lambda x: x[2], reverse=True)
        return similar_pairs[:top_n]
    
    def analyze_contextual_similarity(self, resume_text: str, job_text: str) -> dict:
        """
        Analyze contextual similarity using BERT embeddings.
        
        Args:
            resume_text (str): Resume text
            job_text (str): Job description text
            
        Returns:
            dict: Contextual similarity analysis
        """
        # Calculate overall similarity
        overall_similarity = self.calculate_similarity(resume_text, job_text)
        
        # Split into sentences for detailed analysis
        resume_sentences = [sent.strip() for sent in resume_text.split('.') if sent.strip()]
        job_sentences = [sent.strip() for sent in job_text.split('.') if sent.strip()]
        
        # Find similar sentence pairs
        similar_pairs = self.find_most_similar_sentences(
            resume_sentences, job_sentences, top_n=3
        )
        
        # Calculate average sentence similarity
        if similar_pairs:
            avg_sentence_similarity = np.mean([pair[2] for pair in similar_pairs])
        else:
            avg_sentence_similarity = 0.0
        
        return {
            'overall_similarity': overall_similarity,
            'average_sentence_similarity': float(avg_sentence_similarity),
            'top_similar_pairs': [
                {
                    'resume_sentence': pair[0][:100] + '...' if len(pair[0]) > 100 else pair[0],
                    'job_sentence': pair[1][:100] + '...' if len(pair[1]) > 100 else pair[1],
                    'similarity': float(pair[2])
                }
                for pair in similar_pairs
            ],
            'model_used': self.model_name
        }