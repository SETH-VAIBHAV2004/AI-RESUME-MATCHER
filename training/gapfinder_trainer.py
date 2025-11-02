#!/usr/bin/env python3
"""
Enhanced GapFinder-NLP Model Trainer
Trains the custom model on real resume-job matching datasets.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
import os
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class ResumeJobDataset(Dataset):
    """Dataset class for resume-job matching."""
    
    def __init__(self, csv_file: str, tokenizer, max_length: int = 512):
        """
        Initialize dataset.
        
        Args:
            csv_file (str): Path to CSV file
            tokenizer: BERT tokenizer
            max_length (int): Maximum sequence length
        """
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Combine resume and job text
        resume_text = str(row['resume_text'])[:400]  # Limit length
        job_text = str(row['job_text'])[:400]
        
        # Create input text
        input_text = f"Resume: {resume_text} [SEP] Job: {job_text}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(row['label'], dtype=torch.long)
        }

class EnhancedGapFinderNLP(nn.Module):
    """Enhanced GapFinder-NLP model with improved architecture."""
    
    def __init__(self, model_name: str = 'bert-base-uncased', num_classes: int = 2):
        """
        Initialize enhanced model.
        
        Args:
            model_name (str): Base BERT model name
            num_classes (int): Number of output classes
        """
        super(EnhancedGapFinderNLP, self).__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.3)
        
        # Enhanced classifier with multiple layers
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes)
        )
        
        # Gap analysis head
        self.gap_analyzer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 4),  # 4 skill categories
            nn.Sigmoid()
        )
        
        # Compatibility scorer
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, input_ids, attention_mask):
        """Forward pass."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        
        # Main classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        # Gap analysis
        gap_scores = self.gap_analyzer(pooled_output)
        
        # Compatibility score
        compatibility = self.compatibility_scorer(pooled_output)
        
        return {
            'logits': logits,
            'gap_scores': gap_scores,
            'compatibility': compatibility
        }

class GapFinderTrainer:
    """Trainer class for GapFinder-NLP model."""
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        """Initialize trainer."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = model_name
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = EnhancedGapFinderNLP(model_name)
        self.model.to(self.device)
        
        print(f"🤖 Initialized GapFinder-NLP on {self.device}")
    
    def create_data_loaders(self, train_csv: str, val_csv: str, batch_size: int = 8):
        """Create data loaders for training and validation."""
        print("📊 Creating data loaders...")
        
        train_dataset = ResumeJobDataset(train_csv, self.tokenizer)
        val_dataset = ResumeJobDataset(val_csv, self.tokenizer)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0  # Set to 0 for Windows compatibility
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0
        )
        
        print(f"   • Training samples: {len(train_dataset)}")
        print(f"   • Validation samples: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler, criterion):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        predictions = []
        true_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask)
            
            # Calculate loss
            loss = criterion(outputs['logits'], labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            
            # Get predictions
            preds = torch.argmax(outputs['logits'], dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(true_labels, predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader, criterion):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        predictions = []
        true_labels = []
        compatibility_scores = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = criterion(outputs['logits'], labels)
                total_loss += loss.item()
                
                # Get predictions
                preds = torch.argmax(outputs['logits'], dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                compatibility_scores.extend(outputs['compatibility'].cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(true_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            true_labels, predictions, average='weighted'
        )
        
        # Calculate AUC if we have both classes
        auc = 0.0
        try:
            if len(set(true_labels)) > 1:
                auc = roc_auc_score(true_labels, compatibility_scores)
        except:
            pass
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def train(self, train_csv: str, val_csv: str, epochs: int = 3, 
              batch_size: int = 8, learning_rate: float = 2e-5):
        """Train the model."""
        print("🚀 Starting GapFinder-NLP Training")
        print("=" * 50)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_csv, val_csv, batch_size
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_f1 = 0.0
        training_history = []
        
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, optimizer, scheduler, criterion
            )
            
            # Validate
            val_metrics = self.validate(val_loader, criterion)
            
            # Log results
            print(f"   • Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"   • Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"   • Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['auc']:.4f}")
            
            # Save best model
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                self.save_model('models/gapfinder_best.pt')
                print(f"   ✅ New best model saved (F1: {best_f1:.4f})")
            
            # Track history
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'val_f1': val_metrics['f1'],
                'val_auc': val_metrics['auc']
            })
        
        print("=" * 50)
        print(f"🎉 Training completed! Best F1: {best_f1:.4f}")
        
        # Save training history
        self.save_training_history(training_history)
        
        return training_history
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'tokenizer': self.tokenizer,
            'model_name': self.model_name
        }, path)
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.tokenizer = checkpoint['tokenizer']
        self.model_name = checkpoint['model_name']
        
        print(f"✅ Model loaded from {path}")
    
    def save_training_history(self, history: list):
        """Save training history."""
        os.makedirs('models', exist_ok=True)
        
        with open('models/training_history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        print("📊 Training history saved to models/training_history.json")

def main():
    """Main training function."""
    print("🤖 GapFinder-NLP Enhanced Training")
    print("=" * 60)
    
    # Check if datasets exist
    required_files = ['data/train.csv', 'data/val.csv']
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        print("💡 Run dataset integration first: python data_integration/dataset_combiner.py")
        return False
    
    # Initialize trainer
    trainer = GapFinderTrainer()
    
    # Train model
    history = trainer.train(
        train_csv='data/train.csv',
        val_csv='data/val.csv',
        epochs=3,
        batch_size=8,
        learning_rate=2e-5
    )
    
    print("✅ Training completed successfully!")
    print("💡 Model saved to: models/gapfinder_best.pt")
    
    return True

if __name__ == "__main__":
    main()