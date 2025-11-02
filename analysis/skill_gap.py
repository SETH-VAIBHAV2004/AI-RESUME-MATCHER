from typing import Dict, List, Set, Tuple
import numpy as np

class SkillGapAnalyzer:
    def __init__(self):
        """Initialize skill gap analyzer."""
        self.gap_categories = ['technical', 'tools', 'soft', 'other']
        
        # Recommendation templates for different skill categories
        self.recommendation_templates = {
            'technical': [
                "Consider taking an online course in {skill}",
                "Build a project using {skill} to gain hands-on experience",
                "Practice {skill} through coding challenges and tutorials",
                "Join a community or forum focused on {skill}",
                "Read documentation and best practices for {skill}"
            ],
            'tools': [
                "Get certified in {skill} through official training programs",
                "Set up a personal project using {skill}",
                "Follow tutorials and guides for {skill}",
                "Practice {skill} in a sandbox or trial environment",
                "Join user groups or communities for {skill}"
            ],
            'soft': [
                "Develop {skill} through leadership opportunities",
                "Take a course or workshop on {skill}",
                "Practice {skill} in team projects and collaborations",
                "Seek feedback and mentoring to improve {skill}",
                "Read books and articles about {skill}"
            ],
            'other': [
                "Gain experience in {skill} through relevant projects",
                "Consider certification or training in {skill}",
                "Network with professionals who have {skill} expertise",
                "Volunteer for opportunities that require {skill}",
                "Study industry best practices for {skill}"
            ]
        }
    
    def analyze_skill_gaps(self, resume_skills: Dict[str, List[str]], 
                          job_skills: Dict[str, List[str]]) -> Dict[str, Dict]:
        """
        Analyze skill gaps between resume and job requirements.
        
        Args:
            resume_skills (Dict[str, List[str]]): Skills extracted from resume
            job_skills (Dict[str, List[str]]): Skills extracted from job description
            
        Returns:
            Dict[str, Dict]: Comprehensive skill gap analysis
        """
        gap_analysis = {}
        
        for category in self.gap_categories:
            resume_set = set(skill.lower() for skill in resume_skills.get(category, []))
            job_set = set(skill.lower() for skill in job_skills.get(category, []))
            
            # Find matches and gaps
            matched_skills = resume_set.intersection(job_set)
            missing_skills = job_set - resume_set
            extra_skills = resume_set - job_set
            
            # Calculate match percentage for this category
            if job_set:
                match_percentage = len(matched_skills) / len(job_set) * 100
            else:
                match_percentage = 100.0  # No requirements means 100% match
            
            gap_analysis[category] = {
                'matched_skills': list(matched_skills),
                'missing_skills': list(missing_skills),
                'extra_skills': list(extra_skills),
                'match_percentage': round(match_percentage, 2),
                'total_required': len(job_set),
                'total_matched': len(matched_skills),
                'total_missing': len(missing_skills)
            }
        
        return gap_analysis
    
    def calculate_overall_match_score(self, gap_analysis: Dict[str, Dict], 
                                    weights: Dict[str, float] = None) -> float:
        """
        Calculate overall match score based on skill gap analysis.
        
        Args:
            gap_analysis (Dict[str, Dict]): Skill gap analysis results
            weights (Dict[str, float]): Category weights for scoring
            
        Returns:
            float: Overall match score (0-100)
        """
        if weights is None:
            weights = {
                'technical': 0.4,
                'tools': 0.3,
                'soft': 0.2,
                'other': 0.1
            }
        
        weighted_scores = []
        total_weight = 0
        
        for category, weight in weights.items():
            if category in gap_analysis:
                category_score = gap_analysis[category]['match_percentage']
                weighted_scores.append(category_score * weight)
                total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        overall_score = sum(weighted_scores) / total_weight
        return round(overall_score, 2)
    
    def generate_recommendations(self, gap_analysis: Dict[str, Dict], 
                               max_recommendations: int = 8) -> List[Dict[str, str]]:
        """
        Generate personalized recommendations based on skill gaps.
        
        Args:
            gap_analysis (Dict[str, Dict]): Skill gap analysis results
            max_recommendations (int): Maximum number of recommendations
            
        Returns:
            List[Dict[str, str]]: List of recommendations with category and text
        """
        recommendations = []
        
        # Prioritize categories by gap size and importance
        category_priorities = {
            'technical': 4,
            'tools': 3,
            'soft': 2,
            'other': 1
        }
        
        # Collect missing skills with priorities
        missing_skills_with_priority = []
        for category, analysis in gap_analysis.items():
            if category in category_priorities:
                priority = category_priorities[category]
                gap_size = analysis['total_missing']
                
                for skill in analysis['missing_skills'][:3]:  # Top 3 missing skills per category
                    missing_skills_with_priority.append({
                        'skill': skill,
                        'category': category,
                        'priority': priority,
                        'gap_size': gap_size
                    })
        
        # Sort by priority and gap size
        missing_skills_with_priority.sort(
            key=lambda x: (x['priority'], x['gap_size']), 
            reverse=True
        )
        
        # Generate recommendations
        for item in missing_skills_with_priority[:max_recommendations]:
            skill = item['skill']
            category = item['category']
            
            # Select recommendation template
            templates = self.recommendation_templates.get(category, [])
            if templates:
                template = np.random.choice(templates)
                recommendation_text = template.format(skill=skill.title())
                
                recommendations.append({
                    'category': category,
                    'skill': skill.title(),
                    'recommendation': recommendation_text,
                    'priority': item['priority']
                })
        
        return recommendations
    
    def create_skill_comparison_matrix(self, resume_skills: Dict[str, List[str]], 
                                     job_skills: Dict[str, List[str]]) -> Dict[str, Dict[str, List[str]]]:
        """
        Create a detailed comparison matrix of skills.
        
        Args:
            resume_skills (Dict[str, List[str]]): Skills from resume
            job_skills (Dict[str, List[str]]): Skills from job description
            
        Returns:
            Dict[str, Dict[str, List[str]]]: Detailed skill comparison
        """
        comparison_matrix = {}
        
        for category in self.gap_categories:
            resume_set = set(skill.lower() for skill in resume_skills.get(category, []))
            job_set = set(skill.lower() for skill in job_skills.get(category, []))
            
            comparison_matrix[category] = {
                'resume_only': list(resume_set - job_set),
                'job_only': list(job_set - resume_set),
                'common': list(resume_set.intersection(job_set)),
                'resume_total': list(resume_set),
                'job_total': list(job_set)
            }
        
        return comparison_matrix
    
    def get_gap_insights(self, gap_analysis: Dict[str, Dict]) -> Dict[str, str]:
        """
        Generate insights about skill gaps.
        
        Args:
            gap_analysis (Dict[str, Dict]): Skill gap analysis results
            
        Returns:
            Dict[str, str]: Insights about the gaps
        """
        insights = {}
        
        # Overall assessment
        total_required = sum(analysis['total_required'] for analysis in gap_analysis.values())
        total_matched = sum(analysis['total_matched'] for analysis in gap_analysis.values())
        
        if total_required == 0:
            insights['overall'] = "No specific skill requirements found in the job description."
        else:
            match_rate = (total_matched / total_required) * 100
            if match_rate >= 80:
                insights['overall'] = f"Excellent match! You have {match_rate:.1f}% of required skills."
            elif match_rate >= 60:
                insights['overall'] = f"Good match with {match_rate:.1f}% of required skills. Some gaps to address."
            elif match_rate >= 40:
                insights['overall'] = f"Moderate match with {match_rate:.1f}% of required skills. Significant skill development needed."
            else:
                insights['overall'] = f"Limited match with {match_rate:.1f}% of required skills. Consider extensive skill development."
        
        # Category-specific insights
        for category, analysis in gap_analysis.items():
            missing_count = analysis['total_missing']
            match_percentage = analysis['match_percentage']
            
            if missing_count == 0:
                insights[category] = f"Perfect {category} skills match!"
            elif match_percentage >= 70:
                insights[category] = f"Strong {category} skills with minor gaps ({missing_count} missing)."
            elif match_percentage >= 40:
                insights[category] = f"Moderate {category} skills gap ({missing_count} missing skills)."
            else:
                insights[category] = f"Significant {category} skills gap ({missing_count} missing skills)."
        
        return insights
    
    def prioritize_skill_development(self, gap_analysis: Dict[str, Dict]) -> List[Dict[str, any]]:
        """
        Prioritize skills for development based on gap analysis.
        
        Args:
            gap_analysis (Dict[str, Dict]): Skill gap analysis results
            
        Returns:
            List[Dict[str, any]]: Prioritized skills for development
        """
        development_priorities = []
        
        # Weight categories by importance
        category_weights = {
            'technical': 0.4,
            'tools': 0.3,
            'soft': 0.2,
            'other': 0.1
        }
        
        for category, analysis in gap_analysis.items():
            if category in category_weights:
                weight = category_weights[category]
                missing_skills = analysis['missing_skills']
                
                for skill in missing_skills:
                    priority_score = weight * (1 + len(missing_skills) * 0.1)  # More missing = higher priority
                    
                    development_priorities.append({
                        'skill': skill.title(),
                        'category': category,
                        'priority_score': priority_score,
                        'urgency': 'High' if priority_score > 0.3 else 'Medium' if priority_score > 0.15 else 'Low'
                    })
        
        # Sort by priority score
        development_priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return development_priorities[:10]  # Top 10 priorities