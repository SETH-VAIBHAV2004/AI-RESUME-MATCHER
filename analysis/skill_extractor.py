import json
import re
from typing import Dict, List, Set, Tuple
from rapidfuzz import fuzz, process
import os

class SkillExtractor:
    def __init__(self, skills_dict_path: str = None):
        """
        Initialize skill extractor with predefined skills dictionary.
        
        Args:
            skills_dict_path (str): Path to skills dictionary JSON file
        """
        if skills_dict_path is None:
            skills_dict_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'skills_dict.json')
        
        self.skills_dict = self._load_skills_dict(skills_dict_path)
        self.all_skills = self._flatten_skills()
        
        # Common words to exclude from skill matching
        self.exclude_words = {
            'experience', 'work', 'job', 'position', 'role', 'responsibility',
            'project', 'team', 'company', 'organization', 'department',
            'summary', 'objective', 'education', 'degree', 'university',
            'college', 'school', 'year', 'month', 'time', 'skill',
            'ability', 'knowledge', 'proficient', 'familiar', 'expert',
            'beginner', 'intermediate', 'advanced', 'strong', 'good',
            'excellent', 'outstanding', 'professional', 'career',
            'development', 'software', 'application', 'system', 'data',
            'business', 'management', 'analysis', 'design', 'implementation'
        }
    
    def _load_skills_dict(self, path: str) -> Dict[str, List[str]]:
        """Load skills dictionary from JSON file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Skills dictionary not found at {path}. Using default skills.")
            return {
                'technical': ['python', 'java', 'javascript', 'sql', 'machine learning'],
                'tools': ['git', 'docker', 'aws', 'linux', 'jenkins'],
                'soft': ['leadership', 'communication', 'teamwork', 'problem solving'],
                'other': ['project management', 'agile', 'scrum', 'testing']
            }
    
    def _flatten_skills(self) -> Set[str]:
        """Flatten all skills into a single set."""
        all_skills = set()
        for category_skills in self.skills_dict.values():
            all_skills.update([skill.lower() for skill in category_skills])
        return all_skills
    
    def preprocess_text_for_extraction(self, text: str) -> str:
        """
        Preprocess text for skill extraction.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Preprocessed text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep dots and hyphens for compound skills
        text = re.sub(r'[^\w\s\.\-]', ' ', text)
        
        return text.strip()
    
    def extract_skills_fuzzy(self, text: str, threshold: int = 80) -> Dict[str, List[Tuple[str, int]]]:
        """
        Extract skills using fuzzy matching.
        
        Args:
            text (str): Input text
            threshold (int): Minimum similarity threshold (0-100)
            
        Returns:
            Dict[str, List[Tuple[str, int]]]: Extracted skills by category with scores
        """
        if not text:
            return {category: [] for category in self.skills_dict.keys()}
        
        preprocessed_text = self.preprocess_text_for_extraction(text)
        words = preprocessed_text.split()
        
        # Create phrases of different lengths for better matching
        phrases = []
        for i in range(len(words)):
            # Single words
            phrases.append(words[i])
            # Two-word phrases
            if i < len(words) - 1:
                phrases.append(f"{words[i]} {words[i+1]}")
            # Three-word phrases
            if i < len(words) - 2:
                phrases.append(f"{words[i]} {words[i+1]} {words[i+2]}")
        
        extracted_skills = {category: [] for category in self.skills_dict.keys()}
        
        for category, skills in self.skills_dict.items():
            for skill in skills:
                skill_lower = skill.lower()
                
                # Skip if skill is in exclude words
                if skill_lower in self.exclude_words:
                    continue
                
                # Find best match in phrases
                best_match = process.extractOne(
                    skill_lower, 
                    phrases, 
                    scorer=fuzz.ratio,
                    score_cutoff=threshold
                )
                
                if best_match:
                    # Handle different return formats from rapidfuzz
                    if len(best_match) == 2:
                        matched_phrase, score = best_match
                    elif len(best_match) == 3:
                        matched_phrase, score, _ = best_match
                    else:
                        matched_phrase, score = best_match[0], best_match[1]
                    extracted_skills[category].append((skill, score))
        
        # Remove duplicates and sort by score
        for category in extracted_skills:
            seen_skills = set()
            unique_skills = []
            for skill, score in extracted_skills[category]:
                if skill.lower() not in seen_skills:
                    seen_skills.add(skill.lower())
                    unique_skills.append((skill, score))
            
            extracted_skills[category] = sorted(unique_skills, key=lambda x: x[1], reverse=True)
        
        return extracted_skills
    
    def extract_skills_exact(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills using exact matching.
        
        Args:
            text (str): Input text
            
        Returns:
            Dict[str, List[str]]: Extracted skills by category
        """
        if not text:
            return {category: [] for category in self.skills_dict.keys()}
        
        preprocessed_text = self.preprocess_text_for_extraction(text)
        extracted_skills = {category: [] for category in self.skills_dict.keys()}
        
        for category, skills in self.skills_dict.items():
            for skill in skills:
                skill_lower = skill.lower()
                
                # Skip if skill is in exclude words
                if skill_lower in self.exclude_words:
                    continue
                
                # Check for exact match (with word boundaries)
                pattern = r'\b' + re.escape(skill_lower) + r'\b'
                if re.search(pattern, preprocessed_text):
                    extracted_skills[category].append(skill)
        
        return extracted_skills
    
    def extract_skills_hybrid(self, text: str, exact_threshold: int = 90, 
                            fuzzy_threshold: int = 75) -> Dict[str, List[Tuple[str, str, int]]]:
        """
        Extract skills using hybrid approach (exact + fuzzy matching).
        
        Args:
            text (str): Input text
            exact_threshold (int): Threshold for considering exact matches
            fuzzy_threshold (int): Threshold for fuzzy matches
            
        Returns:
            Dict[str, List[Tuple[str, str, int]]]: Skills with (skill, match_type, score)
        """
        if not text:
            return {category: [] for category in self.skills_dict.keys()}
        
        # Get exact matches
        exact_skills = self.extract_skills_exact(text)
        
        # Get fuzzy matches
        fuzzy_skills = self.extract_skills_fuzzy(text, fuzzy_threshold)
        
        # Combine results
        combined_skills = {category: [] for category in self.skills_dict.keys()}
        
        for category in self.skills_dict.keys():
            # Add exact matches with high score
            for skill in exact_skills[category]:
                combined_skills[category].append((skill, 'exact', 100))
            
            # Add fuzzy matches that aren't already exact matches
            exact_skill_names = {skill.lower() for skill in exact_skills[category]}
            for skill, score in fuzzy_skills[category]:
                if skill.lower() not in exact_skill_names and score >= fuzzy_threshold:
                    match_type = 'exact' if score >= exact_threshold else 'fuzzy'
                    combined_skills[category].append((skill, match_type, score))
        
        # Sort by score and remove duplicates
        for category in combined_skills:
            seen_skills = set()
            unique_skills = []
            for skill, match_type, score in combined_skills[category]:
                if skill.lower() not in seen_skills:
                    seen_skills.add(skill.lower())
                    unique_skills.append((skill, match_type, score))
            
            combined_skills[category] = sorted(unique_skills, key=lambda x: x[2], reverse=True)
        
        return combined_skills
    
    def get_skill_statistics(self, extracted_skills: Dict[str, List]) -> Dict[str, int]:
        """
        Get statistics about extracted skills.
        
        Args:
            extracted_skills (Dict[str, List]): Extracted skills by category
            
        Returns:
            Dict[str, int]: Skill statistics
        """
        stats = {}
        total_skills = 0
        
        for category, skills in extracted_skills.items():
            count = len(skills)
            stats[f"{category}_count"] = count
            total_skills += count
        
        stats['total_skills'] = total_skills
        stats['categories_with_skills'] = sum(1 for count in stats.values() if count > 0 and 'count' in str(count))
        
        return stats
    
    def format_skills_for_display(self, extracted_skills: Dict[str, List]) -> Dict[str, List[str]]:
        """
        Format extracted skills for display purposes.
        
        Args:
            extracted_skills: Extracted skills (can be with or without scores)
            
        Returns:
            Dict[str, List[str]]: Formatted skills by category
        """
        formatted_skills = {}
        
        for category, skills in extracted_skills.items():
            if not skills:
                formatted_skills[category] = []
                continue
            
            # Handle different skill formats
            if isinstance(skills[0], tuple):
                # Skills with scores/match types
                if len(skills[0]) == 2:  # (skill, score)
                    formatted_skills[category] = [skill for skill, _ in skills]
                elif len(skills[0]) == 3:  # (skill, match_type, score)
                    formatted_skills[category] = [skill for skill, _, _ in skills]
            else:
                # Simple skill list
                formatted_skills[category] = skills
        
        return formatted_skills