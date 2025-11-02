import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class MatchingMetrics:
    def __init__(self):
        """Initialize matching metrics calculator."""
        self.metrics_history = []
    
    def calculate_similarity_metrics(self, tfidf_score: float, word2vec_score: float, 
                                   bert_score: float, gapfinder_score: float) -> Dict[str, float]:
        """
        Calculate combined similarity metrics from different models.
        
        Args:
            tfidf_score (float): TF-IDF similarity score
            word2vec_score (float): Word2Vec similarity score
            bert_score (float): BERT similarity score
            gapfinder_score (float): GapFinder-NLP score
            
        Returns:
            Dict[str, float]: Combined metrics
        """
        # Weighted combination
        weights = {
            'tfidf': 0.25,
            'word2vec': 0.25,
            'bert': 0.25,
            'gapfinder': 0.25
        }
        
        final_score = (
            weights['tfidf'] * tfidf_score +
            weights['word2vec'] * word2vec_score +
            weights['bert'] * bert_score +
            weights['gapfinder'] * gapfinder_score
        )
        
        # Calculate additional metrics
        scores = [tfidf_score, word2vec_score, bert_score, gapfinder_score]
        
        metrics = {
            'final_score': round(final_score * 100, 2),  # Convert to percentage
            'tfidf_score': round(tfidf_score * 100, 2),
            'word2vec_score': round(word2vec_score * 100, 2),
            'bert_score': round(bert_score * 100, 2),
            'gapfinder_score': round(gapfinder_score * 100, 2),
            'average_score': round(np.mean(scores) * 100, 2),
            'max_score': round(np.max(scores) * 100, 2),
            'min_score': round(np.min(scores) * 100, 2),
            'score_variance': round(np.var(scores), 4),
            'score_std': round(np.std(scores), 4)
        }
        
        return metrics
    
    def evaluate_skill_matching(self, matched_skills: Dict[str, List[str]], 
                              missing_skills: Dict[str, List[str]], 
                              job_skills: Dict[str, List[str]]) -> Dict[str, float]:
        """
        Evaluate skill matching performance.
        
        Args:
            matched_skills (Dict[str, List[str]]): Matched skills by category
            missing_skills (Dict[str, List[str]]): Missing skills by category
            job_skills (Dict[str, List[str]]): Required job skills by category
            
        Returns:
            Dict[str, float]: Skill matching metrics
        """
        metrics = {}
        
        for category in ['technical', 'tools', 'soft', 'other']:
            matched = len(matched_skills.get(category, []))
            missing = len(missing_skills.get(category, []))
            required = len(job_skills.get(category, []))
            
            if required > 0:
                precision = matched / (matched + missing) if (matched + missing) > 0 else 0
                recall = matched / required
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                metrics[f'{category}_precision'] = round(precision, 3)
                metrics[f'{category}_recall'] = round(recall, 3)
                metrics[f'{category}_f1'] = round(f1_score, 3)
                metrics[f'{category}_match_rate'] = round((matched / required) * 100, 2)
            else:
                metrics[f'{category}_precision'] = 1.0
                metrics[f'{category}_recall'] = 1.0
                metrics[f'{category}_f1'] = 1.0
                metrics[f'{category}_match_rate'] = 100.0
        
        # Overall metrics
        total_matched = sum(len(skills) for skills in matched_skills.values())
        total_missing = sum(len(skills) for skills in missing_skills.values())
        total_required = sum(len(skills) for skills in job_skills.values())
        
        if total_required > 0:
            overall_precision = total_matched / (total_matched + total_missing) if (total_matched + total_missing) > 0 else 0
            overall_recall = total_matched / total_required
            overall_f1 = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
            
            metrics['overall_precision'] = round(overall_precision, 3)
            metrics['overall_recall'] = round(overall_recall, 3)
            metrics['overall_f1'] = round(overall_f1, 3)
            metrics['overall_match_rate'] = round((total_matched / total_required) * 100, 2)
        else:
            metrics['overall_precision'] = 1.0
            metrics['overall_recall'] = 1.0
            metrics['overall_f1'] = 1.0
            metrics['overall_match_rate'] = 100.0
        
        return metrics
    
    def create_model_comparison_chart(self, metrics: Dict[str, float]) -> go.Figure:
        """
        Create a comparison chart of different model scores.
        
        Args:
            metrics (Dict[str, float]): Model metrics
            
        Returns:
            go.Figure: Plotly figure
        """
        models = ['TF-IDF', 'Word2Vec', 'BERT', 'GapFinder-NLP']
        scores = [
            metrics['tfidf_score'],
            metrics['word2vec_score'],
            metrics['bert_score'],
            metrics['gapfinder_score']
        ]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        
        fig = go.Figure(data=[
            go.Bar(
                x=models,
                y=scores,
                marker_color=colors,
                text=[f'{score:.1f}%' for score in scores],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Model Comparison - Similarity Scores',
            xaxis_title='Models',
            yaxis_title='Similarity Score (%)',
            yaxis=dict(range=[0, 100]),
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_skill_gap_visualization(self, gap_analysis: Dict[str, Dict]) -> go.Figure:
        """
        Create visualization for skill gaps by category.
        
        Args:
            gap_analysis (Dict[str, Dict]): Skill gap analysis
            
        Returns:
            go.Figure: Plotly figure
        """
        categories = list(gap_analysis.keys())
        matched = [gap_analysis[cat]['total_matched'] for cat in categories]
        missing = [gap_analysis[cat]['total_missing'] for cat in categories]
        
        fig = go.Figure(data=[
            go.Bar(name='Matched Skills', x=categories, y=matched, marker_color='#2ECC71'),
            go.Bar(name='Missing Skills', x=categories, y=missing, marker_color='#E74C3C')
        ])
        
        fig.update_layout(
            title='Skill Gap Analysis by Category',
            xaxis_title='Skill Categories',
            yaxis_title='Number of Skills',
            barmode='stack',
            template='plotly_white',
            height=400
        )
        
        return fig
    
    def create_match_percentage_gauge(self, final_score: float) -> go.Figure:
        """
        Create a gauge chart for overall match percentage.
        
        Args:
            final_score (float): Final match score (0-100)
            
        Returns:
            go.Figure: Plotly gauge figure
        """
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = final_score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Overall Match Score"},
            delta = {'reference': 70},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(height=300)
        return fig
    
    def generate_performance_report(self, metrics: Dict[str, float], 
                                  gap_analysis: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            metrics (Dict[str, float]): Model metrics
            gap_analysis (Dict[str, Dict]): Skill gap analysis
            
        Returns:
            Dict[str, Any]: Performance report
        """
        # Calculate performance grade
        final_score = metrics['final_score']
        if final_score >= 90:
            grade = 'A+'
            performance_level = 'Excellent'
        elif final_score >= 80:
            grade = 'A'
            performance_level = 'Very Good'
        elif final_score >= 70:
            grade = 'B'
            performance_level = 'Good'
        elif final_score >= 60:
            grade = 'C'
            performance_level = 'Fair'
        else:
            grade = 'D'
            performance_level = 'Needs Improvement'
        
        # Identify strengths and weaknesses
        model_scores = {
            'TF-IDF': metrics['tfidf_score'],
            'Word2Vec': metrics['word2vec_score'],
            'BERT': metrics['bert_score'],
            'GapFinder-NLP': metrics['gapfinder_score']
        }
        
        best_model = max(model_scores, key=model_scores.get)
        worst_model = min(model_scores, key=model_scores.get)
        
        # Category performance
        category_performance = {}
        for category, analysis in gap_analysis.items():
            match_rate = analysis['match_percentage']
            if match_rate >= 80:
                category_performance[category] = 'Strong'
            elif match_rate >= 60:
                category_performance[category] = 'Moderate'
            else:
                category_performance[category] = 'Weak'
        
        report = {
            'overall_grade': grade,
            'performance_level': performance_level,
            'final_score': final_score,
            'best_performing_model': best_model,
            'worst_performing_model': worst_model,
            'model_consistency': 'High' if metrics['score_std'] < 0.1 else 'Medium' if metrics['score_std'] < 0.2 else 'Low',
            'category_performance': category_performance,
            'total_skills_matched': sum(analysis['total_matched'] for analysis in gap_analysis.values()),
            'total_skills_missing': sum(analysis['total_missing'] for analysis in gap_analysis.values()),
            'recommendations_count': sum(analysis['total_missing'] for analysis in gap_analysis.values()),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return report
    
    def save_metrics_history(self, metrics: Dict[str, float], 
                           gap_analysis: Dict[str, Dict]) -> None:
        """
        Save metrics to history for tracking.
        
        Args:
            metrics (Dict[str, float]): Model metrics
            gap_analysis (Dict[str, Dict]): Skill gap analysis
        """
        history_entry = {
            'timestamp': pd.Timestamp.now(),
            'final_score': metrics['final_score'],
            'tfidf_score': metrics['tfidf_score'],
            'word2vec_score': metrics['word2vec_score'],
            'bert_score': metrics['bert_score'],
            'gapfinder_score': metrics['gapfinder_score'],
            'total_matched': sum(analysis['total_matched'] for analysis in gap_analysis.values()),
            'total_missing': sum(analysis['total_missing'] for analysis in gap_analysis.values())
        }
        
        self.metrics_history.append(history_entry)
    
    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get summary statistics of metrics history.
        
        Returns:
            Dict[str, float]: Summary statistics
        """
        if not self.metrics_history:
            return {}
        
        df = pd.DataFrame(self.metrics_history)
        
        summary = {
            'average_final_score': df['final_score'].mean(),
            'max_final_score': df['final_score'].max(),
            'min_final_score': df['final_score'].min(),
            'score_improvement_trend': df['final_score'].iloc[-1] - df['final_score'].iloc[0] if len(df) > 1 else 0,
            'total_evaluations': len(df)
        }
        
        return summary