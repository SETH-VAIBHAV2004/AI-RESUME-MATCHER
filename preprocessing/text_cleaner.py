import re
import spacy
from typing import List, Tuple
import string

class TextCleaner:
    def __init__(self):
        """Initialize the text cleaner with spaCy model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy English model...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
    
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing unwanted characters and formatting.
        
        Args:
            text (str): Raw text to clean
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers
        text = re.sub(r'[\+]?[1-9]?[0-9]{3}-?[0-9]{3}-?[0-9]{4}', '', text)
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?]', ' ', text)
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text using spaCy.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of lemmatized tokens
        """
        if not text:
            return []
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract lemmatized tokens, excluding stop words and punctuation
        tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop 
            and not token.is_punct 
            and not token.is_space
            and len(token.text) > 2
            and token.text not in string.punctuation
        ]
        
        return tokens
    
    def extract_sentences(self, text: str) -> List[str]:
        """
        Extract sentences from text.
        
        Args:
            text (str): Text to process
            
        Returns:
            List[str]: List of sentences
        """
        if not text:
            return []
        
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]
        
        return sentences
    
    def preprocess_for_matching(self, text: str) -> Tuple[str, List[str]]:
        """
        Complete preprocessing pipeline for text matching.
        
        Args:
            text (str): Raw text to preprocess
            
        Returns:
            Tuple[str, List[str]]: Cleaned text and tokenized/lemmatized words
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_and_lemmatize(cleaned_text)
        
        return cleaned_text, tokens
    
    def remove_common_resume_words(self, tokens: List[str]) -> List[str]:
        """
        Remove common resume words that don't add value to skill matching.
        
        Args:
            tokens (List[str]): List of tokens
            
        Returns:
            List[str]: Filtered tokens
        """
        common_words = {
            'experience', 'work', 'job', 'position', 'role', 'responsibility',
            'project', 'team', 'company', 'organization', 'department',
            'summary', 'objective', 'education', 'degree', 'university',
            'college', 'school', 'year', 'month', 'time', 'skill',
            'ability', 'knowledge', 'proficient', 'familiar', 'expert',
            'beginner', 'intermediate', 'advanced', 'strong', 'good',
            'excellent', 'outstanding', 'professional', 'career'
        }
        
        return [token for token in tokens if token not in common_words]