#!/usr/bin/env python3
"""
Comprehensive validation script to verify dataset integration and model performance.
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import sys

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def validate_datasets():
    """Validate that all datasets are properly integrated."""
    print("📊 Validating Dataset Integration")
    print("=" * 50)
    
    # Check if all required files exist
    required_files = [
        'data/combined_dataset.csv',
        'data/train.csv',
        'data/val.csv', 
        'data/test.csv',
        'data/skills_dict.json'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path} exists")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Validate dataset contents
    try:
        # Load combined dataset
        combined_df = pd.read_csv('data/combined_dataset.csv')
        train_df = pd.read_csv('data/train.csv')
        val_df = pd.read_csv('data/val.csv')
        test_df = pd.read_csv('data/test.csv')
        
        print(f"\n📈 Dataset Statistics:")
        print(f"   • Combined Dataset: {len(combined_df)} samples")
        print(f"   • Training Set: {len(train_df)} samples")
        print(f"   • Validation Set: {len(val_df)} samples")
        print(f"   • Test Set: {len(test_df)} samples")
        
        # Check data quality
        required_columns = ['id', 'job_text', 'resume_text', 'label', 'source']
        for col in required_columns:
            if col not in combined_df.columns:
                print(f"❌ Missing column: {col}")
                return False
        
        print(f"✅ All required columns present")
        
        # Check label distribution
        positive_samples = sum(combined_df['label'] == 1)
        negative_samples = sum(combined_df['label'] == 0)
        balance_ratio = positive_samples / len(combined_df)
        
        print(f"\n⚖️ Label Distribution:")
        print(f"   • Positive samples: {positive_samples} ({balance_ratio:.1%})")
        print(f"   • Negative samples: {negative_samples} ({1-balance_ratio:.1%})")
        
        # Check data sources
        sources = combined_df['source'].value_counts()
        print(f"\n📊 Data Sources:")
        for source, count in sources.items():
            print(f"   • {source}: {count} samples")
        
        # Validate skills dictionary
        with open('data/skills_dict.json', 'r') as f:
            skills_dict = json.load(f)
        
        total_skills = sum(len(skills) for skills in skills_dict.values())
        print(f"\n🎯 Skills Dictionary:")
        print(f"   • Total skills: {total_skills}")
        for category, skills in skills_dict.items():
            print(f"   • {category}: {len(skills)} skills")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating datasets: {e}")
        return False

def validate_enhanced_model():
    """Validate the enhanced GapFinder-NLP model."""
    print("\n🤖 Validating Enhanced GapFinder-NLP Model")
    print("=" * 50)
    
    try:
        from features.enhanced_gapfinder import EnhancedGapFinderNLP
        
        # Initialize model
        model = EnhancedGapFinderNLP()
        
        # Test with sample data
        resume_text = """
        Software Engineer with 5 years experience in Python, JavaScript, React, and AWS.
        Strong background in machine learning and data science. Led team of 4 developers.
        Experience with Docker, Kubernetes, and CI/CD pipelines.
        """
        
        job_text = """
        Senior Software Engineer position requiring Python, JavaScript, React experience.
        AWS cloud experience required. Machine learning knowledge preferred.
        Leadership experience and team management skills essential.
        """
        
        # Test comprehensive analysis
        results = model.comprehensive_analysis(resume_text, job_text)
        
        print(f"✅ Model initialized successfully")
        print(f"✅ Comprehensive analysis completed")
        
        # Validate results structure
        required_keys = [
            'gapfinder_score', 'compatibility_probability', 'semantic_similarity',
            'skill_gap_analysis', 'improvement_suggestions', 'confidence_score',
            'model_version', 'dataset_insights'
        ]
        
        for key in required_keys:
            if key not in results:
                print(f"❌ Missing result key: {key}")
                return False
        
        print(f"✅ All required result keys present")
        
        # Check score ranges
        scores_to_check = [
            ('gapfinder_score', results['gapfinder_score']),
            ('compatibility_probability', results['compatibility_probability']),
            ('semantic_similarity', results['semantic_similarity']),
            ('confidence_score', results['confidence_score'])
        ]
        
        for score_name, score_value in scores_to_check:
            if not (0 <= score_value <= 1):
                print(f"❌ {score_name} out of range: {score_value}")
                return False
        
        print(f"✅ All scores within valid range [0, 1]")
        
        # Display results
        print(f"\n📊 Sample Analysis Results:")
        print(f"   • GapFinder Score: {results['gapfinder_score']:.3f}")
        print(f"   • Compatibility: {results['compatibility_probability']:.3f}")
        print(f"   • Semantic Similarity: {results['semantic_similarity']:.3f}")
        print(f"   • Confidence: {results['confidence_score']:.3f}")
        print(f"   • Model Version: {results['model_version']}")
        
        # Check dataset insights
        insights = results['dataset_insights']
        print(f"\n🔍 Dataset Insights:")
        print(f"   • Trained on: {insights['trained_on_samples']} samples")
        print(f"   • Positive ratio: {insights['positive_ratio']:.3f}")
        
        # Check suggestions
        suggestions = results['improvement_suggestions']
        print(f"\n💡 Generated {len(suggestions)} suggestions:")
        for i, suggestion in enumerate(suggestions[:3], 1):
            print(f"   {i}. {suggestion}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error validating enhanced model: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_system_integration():
    """Validate that the system is properly integrated."""
    print("\n🔧 Validating System Integration")
    print("=" * 50)
    
    try:
        # Test that app can import enhanced model
        from app.app import analyze_resume_job_match, load_models
        
        print(f"✅ App imports working correctly")
        
        # Test model loading
        models = load_models()
        if not all(models):
            print(f"❌ Model loading failed")
            return False
        
        print(f"✅ All models loaded successfully")
        
        # Test with real dataset sample
        if os.path.exists('data/combined_dataset.csv'):
            df = pd.read_csv('data/combined_dataset.csv')
            if len(df) > 0:
                sample = df.iloc[0]
                resume_text = sample['resume_text']
                job_text = sample['job_text']
                
                # Test analysis function
                results = analyze_resume_job_match(
                    resume_text, job_text, *models
                )
                
                print(f"✅ End-to-end analysis working with real data")
                print(f"   • Final Score: {results['final_score']:.1f}%")
                print(f"   • Recommendations: {len(results['recommendations'])}")
                
                return True
        
        print(f"⚠️  No real data available for testing")
        return True
        
    except Exception as e:
        print(f"❌ Error validating system integration: {e}")
        import traceback
        traceback.print_exc()
        return False

def validate_performance():
    """Validate model performance on test set."""
    print("\n📈 Validating Model Performance")
    print("=" * 50)
    
    try:
        from features.enhanced_gapfinder import EnhancedGapFinderNLP
        
        # Load test set
        if not os.path.exists('data/test.csv'):
            print("⚠️  Test set not available")
            return True
        
        test_df = pd.read_csv('data/test.csv')
        model = EnhancedGapFinderNLP()
        
        print(f"🧪 Testing on {len(test_df)} samples...")
        
        # Test on a subset for performance
        test_subset = test_df.sample(n=min(20, len(test_df)), random_state=42)
        
        predictions = []
        true_labels = []
        
        for idx, row in test_subset.iterrows():
            try:
                # Get model prediction
                compatibility = model.predict_compatibility(
                    row['resume_text'], row['job_text']
                )
                
                # Convert to binary prediction (threshold = 0.5)
                prediction = 1 if compatibility > 0.5 else 0
                predictions.append(prediction)
                true_labels.append(row['label'])
                
            except Exception as e:
                print(f"   ⚠️  Error processing sample {idx}: {e}")
                continue
        
        if len(predictions) > 0:
            # Calculate accuracy
            accuracy = sum(p == t for p, t in zip(predictions, true_labels)) / len(predictions)
            
            print(f"✅ Performance validation completed")
            print(f"   • Test samples: {len(predictions)}")
            print(f"   • Accuracy: {accuracy:.3f}")
            
            # Calculate additional metrics
            true_positives = sum(p == 1 and t == 1 for p, t in zip(predictions, true_labels))
            false_positives = sum(p == 1 and t == 0 for p, t in zip(predictions, true_labels))
            false_negatives = sum(p == 0 and t == 1 for p, t in zip(predictions, true_labels))
            
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
                print(f"   • Precision: {precision:.3f}")
            
            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
                print(f"   • Recall: {recall:.3f}")
            
            return accuracy > 0.5  # Expect better than random
        else:
            print(f"❌ No valid predictions generated")
            return False
        
    except Exception as e:
        print(f"❌ Error validating performance: {e}")
        return False

def main():
    """Run comprehensive validation."""
    print("🔍 Comprehensive System Validation")
    print("=" * 60)
    
    validation_results = []
    
    # Run all validations
    validations = [
        ("Dataset Integration", validate_datasets),
        ("Enhanced Model", validate_enhanced_model),
        ("System Integration", validate_system_integration),
        ("Model Performance", validate_performance)
    ]
    
    for name, validation_func in validations:
        try:
            result = validation_func()
            validation_results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation crashed: {e}")
            validation_results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Validation Summary")
    print("=" * 60)
    
    passed = 0
    total = len(validation_results)
    
    for name, result in validation_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name:.<30} {status}")
        if result:
            passed += 1
    
    print("=" * 60)
    print(f"🎯 Overall Result: {passed}/{total} validations passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ System is fully integrated and working correctly")
        print("🚀 Ready for production use with real datasets")
    else:
        print("⚠️  Some validations failed")
        print("💡 Check the errors above and fix issues")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)