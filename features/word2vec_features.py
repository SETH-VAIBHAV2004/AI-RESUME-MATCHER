import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

class Word2VecFeatures:
    def __init__(self, vector_size: int = 100, window: int = 5, min_count: int = 1, workers: int = 4):
        """
        Initialize Word2Vec feature extractor.
        
        Args:
            vector_size (int): Dimensionality of word vectors
            window (int): Maximum distance between current and predicted word
            min_count (int): Minimum word frequency
            workers (int): Number of worker threads
        """
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model = None
        self.is_trained = False
    
    def train_model(self, tokenized_texts: List[List[str]]) -> None:
        """
        Train Word2Vec model on tokenized texts.
        
        Args:
            tokenized_texts (List[List[str]]): List of tokenized text documents
        """
        if not tokenized_texts or not any(tokenized_texts):
            raise ValueError("No valid tokenized texts provided for training")
        
        self.model = Word2Vec(
            sentences=tokenized_texts,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
            epochs=10
        )
        self.is_trained = True
    
    def get_document_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Get document vector by averaging word vectors.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            np.ndarray: Document vector
        """
        if not self.is_trained or not self.model:
            raise ValueError("Model must be trained before getting document vectors")
        
        if not tokens:
            return np.zeros(self.vector_size)
        
        # Get vectors for tokens that exist in vocabulary
        vectors = []
        for token in tokens:
            if token in self.model.wv:
                vectors.append(self.model.wv[token])
        
        if not vectors:
            return np.zeros(self.vector_size)
        
        # Return average of word vectors
        return np.mean(vectors, axis=0)
    
    def calculate_similarity(self, resume_tokens: List[str], job_tokens: List[str]) -> float:
        """
        Calculate Word2Vec cosine similarity between resume and job description.
        
        Args:
            resume_tokens (List[str]): Resume tokens
            job_tokens (List[str]): Job description tokens
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Train model on combined tokens
        all_tokens = [resume_tokens, job_tokens]
        self.train_model(all_tokens)
        
        # Get document vectors
        resume_vector = self.get_document_vector(resume_tokens)
        job_vector = self.get_document_vector(job_tokens)
        
        # Calculate cosine similarity
        if np.all(resume_vector == 0) or np.all(job_vector == 0):
            return 0.0
        
        similarity = cosine_similarity(
            resume_vector.reshape(1, -1), 
            job_vector.reshape(1, -1)
        )[0, 0]
        
        return float(similarity)
    
    def find_similar_words(self, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Find words most similar to the given word.
        
        Args:
            word (str): Target word
            top_n (int): Number of similar words to return
            
        Returns:
            List[Tuple[str, float]]: List of (word, similarity) tuples
        """
        if not self.is_trained or not self.model:
            raise ValueError("Model must be trained before finding similar words")
        
        if word not in self.model.wv:
            return []
        
        try:
            similar_words = self.model.wv.most_similar(word, topn=top_n)
            return similar_words
        except KeyError:
            return []
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get vector representation of a word.
        
        Args:
            word (str): Target word
            
        Returns:
            np.ndarray: Word vector
        """
        if not self.is_trained or not self.model:
            raise ValueError("Model must be trained before getting word vectors")
        
        if word in self.model.wv:
            return self.model.wv[word]
        else:
            return np.zeros(self.vector_size)
    
    def analyze_semantic_similarity(self, resume_tokens: List[str], job_tokens: List[str]) -> dict:
        """
        Analyze semantic similarity between resume and job description.
        
        Args:
            resume_tokens (List[str]): Resume tokens
            job_tokens (List[str]): Job description tokens
            
        Returns:
            dict: Semantic similarity analysis
        """
        # Calculate overall similarity
        similarity_score = self.calculate_similarity(resume_tokens, job_tokens)
        
        # Find semantic matches
        semantic_matches = []
        for resume_word in set(resume_tokens):
            if resume_word in self.model.wv:
                for job_word in set(job_tokens):
                    if job_word in self.model.wv and resume_word != job_word:
                        try:
                            word_similarity = self.model.wv.similarity(resume_word, job_word)
                            if word_similarity > 0.6:  # Threshold for semantic similarity
                                semantic_matches.append({
                                    'resume_word': resume_word,
                                    'job_word': job_word,
                                    'similarity': word_similarity
                                })
                        except KeyError:
                            continue
        
        # Sort by similarity
        semantic_matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'overall_similarity': similarity_score,
            'semantic_matches': semantic_matches[:10],  # Top 10 matches
            'vocabulary_size': len(self.model.wv) if self.model else 0
        }