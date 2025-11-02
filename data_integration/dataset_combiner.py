#!/usr/bin/env python3
"""
Dataset Integration Script for Resume-Job Matcher
Combines multiple real datasets into a unified format for training.
"""

import pandas as pd
import numpy as np
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class DatasetCombiner:
    def __init__(self, datasets_dir: str = "Datasets/data", output_dir: str = "data"):
        """
        Initialize dataset combiner.
        
        Args:
            datasets_dir (str): Directory containing raw datasets
            output_dir (str): Directory for processed datasets
        """
        self.datasets_dir = Path(datasets_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Ensure we have the datasets
        self.dataset_files = {
            'resume_score_details': self.datasets_dir / 'resume_score_details.csv',
            'resume_job_fit': self.datasets_dir / 'resume_job_fit.csv',
            'resume_jd_match': self.datasets_dir / 'resume_jd_match.csv',
            'job_dataset': self.datasets_dir / 'job_dataset.csv'
        }
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text data."""
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Remove extra whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)\[\]\{\}]', ' ', text)
        
        return text.strip()
    
    def extract_score_from_complex_data(self, score_data: str) -> float:
        """Extract numerical score from complex score data."""
        if pd.isna(score_data) or not isinstance(score_data, str):
            return 0.5  # Default neutral score
        
        try:
            # Try to find numerical scores in the data
            scores = re.findall(r'score[\'\"]*:\s*(\d+(?:\.\d+)?)', score_data.lower())
            if scores:
                # Average all found scores and normalize to 0-1
                avg_score = np.mean([float(s) for s in scores])
                return min(avg_score / 10.0, 1.0)  # Assuming scores are 0-10
            
            # Look for percentage matches
            percentages = re.findall(r'(\d+(?:\.\d+)?)%', score_data)
            if percentages:
                avg_pct = np.mean([float(p) for p in percentages])
                return avg_pct / 100.0
            
            # Look for "meets" criteria
            meets_count = score_data.lower().count('true')
            total_criteria = score_data.lower().count('meets')
            if total_criteria > 0:
                return meets_count / total_criteria
            
            return 0.5  # Default if no clear score found
            
        except Exception:
            return 0.5
    
    def process_resume_score_details(self) -> pd.DataFrame:
        """Process the resume_score_details.csv dataset."""
        print("📊 Processing resume_score_details.csv...")
        
        try:
            df = pd.read_csv(self.dataset_files['resume_score_details'])
            print(f"   • Loaded {len(df)} records")
            
            processed_data = []
            
            for idx, row in df.iterrows():
                # Extract job description and resume
                job_text = self.clean_text(str(row.get('input_job_description', '')))
                resume_text = self.clean_text(str(row.get('input_resume', '')))
                
                # Extract score from output_scores
                score_data = str(row.get('output_scores', ''))
                match_score = self.extract_score_from_complex_data(score_data)
                
                # Create binary label (1 if score > 0.6, 0 otherwise)
                label = 1 if match_score > 0.6 else 0
                
                if len(job_text) > 50 and len(resume_text) > 50:  # Valid entries
                    processed_data.append({
                        'id': f'rsd_{idx}',
                        'job_text': job_text,
                        'resume_text': resume_text,
                        'match_score': match_score,
                        'label': label,
                        'source': 'resume_score_details'
                    })
            
            result_df = pd.DataFrame(processed_data)
            print(f"   • Processed {len(result_df)} valid records")
            return result_df
            
        except Exception as e:
            print(f"   ❌ Error processing resume_score_details: {e}")
            return pd.DataFrame()
    
    def process_resume_job_fit(self) -> pd.DataFrame:
        """Process the resume_job_fit.csv dataset."""
        print("📊 Processing resume_job_fit.csv...")
        
        try:
            # Read in chunks due to large size
            chunk_size = 1000
            processed_data = []
            
            for chunk_idx, chunk in enumerate(pd.read_csv(self.dataset_files['resume_job_fit'], chunksize=chunk_size)):
                print(f"   • Processing chunk {chunk_idx + 1}...")
                
                for idx, row in chunk.iterrows():
                    # Extract data based on actual column structure
                    resume_text = self.clean_text(str(row.get('resume_text', '')))
                    job_text = self.clean_text(str(row.get('job_description_text', '')))
                    
                    # Process label - "Fit" = 1, "No Fit" = 0
                    label_val = str(row.get('label', '')).strip().lower()
                    label = 1 if 'fit' in label_val and 'no' not in label_val else 0
                    
                    if len(job_text) > 50 and len(resume_text) > 50:
                        processed_data.append({
                            'id': f'rjf_{chunk_idx}_{idx}',
                            'job_text': job_text,
                            'resume_text': resume_text,
                            'match_score': float(label),
                            'label': label,
                            'source': 'resume_job_fit'
                        })
                
                # Limit processing to avoid memory issues
                if len(processed_data) > 3000:
                    break
            
            result_df = pd.DataFrame(processed_data)
            print(f"   • Processed {len(result_df)} valid records")
            return result_df
            
        except Exception as e:
            print(f"   ❌ Error processing resume_job_fit: {e}")
            return pd.DataFrame()
    
    def process_resume_jd_match(self) -> pd.DataFrame:
        """Process the resume_jd_match.csv dataset."""
        print("📊 Processing resume_jd_match.csv...")
        
        try:
            # Read in chunks due to large size
            chunk_size = 1000
            processed_data = []
            
            for chunk_idx, chunk in enumerate(pd.read_csv(self.dataset_files['resume_jd_match'], chunksize=chunk_size)):
                print(f"   • Processing chunk {chunk_idx + 1}...")
                
                for idx, row in chunk.iterrows():
                    # Extract data from the 'text' column which contains combined job+resume
                    text_content = str(row.get('text', ''))
                    
                    # Parse the combined text format: "For the given job description <<JOB>> and resume <<RESUME>>"
                    job_text = ""
                    resume_text = ""
                    
                    # Split by "and resume" to separate job and resume parts
                    if 'and resume' in text_content:
                        parts = text_content.split('and resume', 1)
                        
                        # Extract job description (after "job description <<" and before ">>")
                        job_part = parts[0]
                        job_start = job_part.find('<<')
                        if job_start != -1:
                            job_end = job_part.find('>>', job_start)
                            if job_end != -1:
                                job_text = self.clean_text(job_part[job_start+2:job_end])
                        
                        # Extract resume (after "<<" in resume part and before ">>")
                        if len(parts) > 1:
                            resume_part = parts[1]
                            resume_start = resume_part.find('<<')
                            if resume_start != -1:
                                resume_end = resume_part.find('>>', resume_start)
                                if resume_end != -1:
                                    resume_text = self.clean_text(resume_part[resume_start+2:resume_end])
                    else:
                        # Fallback: try to extract any text between << >>
                        matches = re.findall(r'<<(.+?)>>', text_content, re.DOTALL)
                        if len(matches) >= 2:
                            job_text = self.clean_text(matches[0])
                            resume_text = self.clean_text(matches[1])
                        elif len(matches) == 1:
                            # If only one match, assume it's the job description
                            job_text = self.clean_text(matches[0])
                            resume_text = ""
                    
                    # Process label - "Fit" = 1, "No Fit" = 0
                    label_val = str(row.get('label', '')).strip().lower()
                    label = 1 if 'fit' in label_val and 'no' not in label_val else 0
                    
                    if len(job_text) > 50 and len(resume_text) > 50:
                        processed_data.append({
                            'id': f'rjm_{chunk_idx}_{idx}',
                            'job_text': job_text,
                            'resume_text': resume_text,
                            'match_score': float(label),
                            'label': label,
                            'source': 'resume_jd_match'
                        })
                
                # Limit processing
                if len(processed_data) > 3000:
                    break
            
            result_df = pd.DataFrame(processed_data)
            print(f"   • Processed {len(result_df)} valid records")
            return result_df
            
        except Exception as e:
            print(f"   ❌ Error processing resume_jd_match: {e}")
            return pd.DataFrame()
    
    def extract_skills_from_job_dataset(self) -> List[str]:
        """Extract skills from job_dataset.csv to enhance skill vocabulary."""
        print("🔍 Extracting skills from job_dataset.csv...")
        
        try:
            skills = set()
            chunk_size = 1000
            
            for chunk in pd.read_csv(self.dataset_files['job_dataset'], chunksize=chunk_size):
                for idx, row in chunk.iterrows():
                    # Look for job description or requirements columns
                    for col in chunk.columns:
                        if any(keyword in col.lower() for keyword in ['description', 'requirement', 'skill', 'qualification']):
                            text = str(row[col]).lower()
                            
                            # Extract common skill patterns
                            skill_patterns = [
                                r'\b(python|java|javascript|react|angular|vue|node\.?js|express|django|flask)\b',
                                r'\b(aws|azure|gcp|docker|kubernetes|jenkins|git|linux|windows)\b',
                                r'\b(sql|mysql|postgresql|mongodb|redis|elasticsearch)\b',
                                r'\b(machine learning|deep learning|ai|data science|analytics)\b',
                                r'\b(html|css|bootstrap|tailwind|sass|less)\b',
                                r'\b(agile|scrum|kanban|devops|ci/cd)\b'
                            ]
                            
                            for pattern in skill_patterns:
                                matches = re.findall(pattern, text)
                                skills.update(matches)
                
                # Limit processing
                if len(skills) > 200:
                    break
            
            skill_list = list(skills)[:100]  # Top 100 extracted skills
            print(f"   • Extracted {len(skill_list)} unique skills")
            return skill_list
            
        except Exception as e:
            print(f"   ❌ Error extracting skills: {e}")
            return []
    
    def combine_datasets(self) -> pd.DataFrame:
        """Combine all datasets into unified format."""
        print("🔄 Combining datasets...")
        
        combined_data = []
        
        # Process each dataset
        datasets = [
            self.process_resume_score_details(),
            self.process_resume_job_fit(),
            self.process_resume_jd_match()
        ]
        
        # Combine all datasets
        for df in datasets:
            if not df.empty:
                combined_data.append(df)
        
        if not combined_data:
            print("❌ No valid data found in any dataset!")
            return pd.DataFrame()
        
        # Concatenate all datasets
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Remove duplicates based on text similarity
        print("🧹 Removing duplicates...")
        combined_df = combined_df.drop_duplicates(subset=['job_text', 'resume_text'])
        
        # Balance the dataset
        print("⚖️ Balancing dataset...")
        positive_samples = combined_df[combined_df['label'] == 1]
        negative_samples = combined_df[combined_df['label'] == 0]
        
        # Balance to have roughly equal positive and negative samples
        min_samples = min(len(positive_samples), len(negative_samples), 2500)
        
        balanced_df = pd.concat([
            positive_samples.sample(n=min_samples, random_state=42),
            negative_samples.sample(n=min_samples, random_state=42)
        ], ignore_index=True)
        
        # Shuffle the dataset
        balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✅ Combined dataset created with {len(balanced_df)} samples")
        print(f"   • Positive samples: {sum(balanced_df['label'] == 1)}")
        print(f"   • Negative samples: {sum(balanced_df['label'] == 0)}")
        
        return balanced_df
    
    def split_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split dataset into train/validation/test sets (70/15/15)."""
        print("📊 Splitting dataset...")
        
        # Shuffle dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        n = len(df)
        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        
        train_df = df[:train_size]
        val_df = df[train_size:train_size + val_size]
        test_df = df[train_size + val_size:]
        
        print(f"   • Training set: {len(train_df)} samples")
        print(f"   • Validation set: {len(val_df)} samples")
        print(f"   • Test set: {len(test_df)} samples")
        
        return train_df, val_df, test_df
    
    def enhance_skills_dict(self, extracted_skills: List[str]):
        """Enhance the skills dictionary with extracted skills."""
        print("🔧 Enhancing skills dictionary...")
        
        skills_dict_path = self.output_dir / 'skills_dict.json'
        
        try:
            # Load existing skills dictionary
            with open(skills_dict_path, 'r') as f:
                skills_dict = json.load(f)
            
            # Add extracted skills to technical category
            existing_technical = set(skill.lower() for skill in skills_dict.get('technical', []))
            
            new_skills = []
            for skill in extracted_skills:
                if skill.lower() not in existing_technical:
                    new_skills.append(skill)
                    skills_dict['technical'].append(skill)
            
            # Save enhanced dictionary
            with open(skills_dict_path, 'w') as f:
                json.dump(skills_dict, f, indent=2)
            
            print(f"   • Added {len(new_skills)} new skills to dictionary")
            
        except Exception as e:
            print(f"   ❌ Error enhancing skills dictionary: {e}")
    
    def run_integration(self):
        """Run the complete dataset integration process."""
        print("🚀 Starting Dataset Integration")
        print("=" * 60)
        
        # Check if datasets exist
        missing_files = []
        for name, path in self.dataset_files.items():
            if not path.exists():
                missing_files.append(name)
        
        if missing_files:
            print(f"❌ Missing dataset files: {missing_files}")
            return False
        
        # Combine datasets
        combined_df = self.combine_datasets()
        
        if combined_df.empty:
            print("❌ Failed to create combined dataset")
            return False
        
        # Split dataset
        train_df, val_df, test_df = self.split_dataset(combined_df)
        
        # Save datasets
        print("💾 Saving datasets...")
        combined_df.to_csv(self.output_dir / 'combined_dataset.csv', index=False)
        train_df.to_csv(self.output_dir / 'train.csv', index=False)
        val_df.to_csv(self.output_dir / 'val.csv', index=False)
        test_df.to_csv(self.output_dir / 'test.csv', index=False)
        
        # Extract and enhance skills
        extracted_skills = self.extract_skills_from_job_dataset()
        if extracted_skills:
            self.enhance_skills_dict(extracted_skills)
        
        print("=" * 60)
        print("🎉 Dataset integration completed successfully!")
        print(f"📁 Files created:")
        print(f"   • combined_dataset.csv ({len(combined_df)} samples)")
        print(f"   • train.csv ({len(train_df)} samples)")
        print(f"   • val.csv ({len(val_df)} samples)")
        print(f"   • test.csv ({len(test_df)} samples)")
        print("=" * 60)
        
        return True

def main():
    """Main function to run dataset integration."""
    combiner = DatasetCombiner()
    success = combiner.run_integration()
    
    if success:
        print("✅ Ready for model training!")
        print("💡 Next step: Run model training with the new datasets")
    else:
        print("❌ Dataset integration failed")
    
    return success

if __name__ == "__main__":
    main()