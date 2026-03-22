"""
Training Script for GcNS (RoBERTa-based Negotiation Strategy Classifier)
Specifically designed for final_agentic_hr_dataset.json format

Usage:
    python train_gcns.py --data_path ../hr_conversations/final_agentic_hr_dataset.json
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import json
from tqdm import tqdm
import os
import argparse


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    model_name: str = "roberta-base"
    max_length: int = 256
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "trained_models/gcns_negotiation_classifier"
    val_split: float = 0.15
    test_split: float = 0.15


class NegotiationStrategyDataset(Dataset):
    """
    Dataset for negotiation strategy classification.
    Handles your specific data format with context awareness.
    """
    
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: RobertaTokenizer,
        strategy_to_id: Dict[str, int],
        max_length: int = 256,
    ):
        self.examples = examples
        self.tokenizer = tokenizer
        self.strategy_to_id = strategy_to_id
        self.max_length = max_length
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        item = self.examples[idx]
        
        # Get text and context
        text = item["text"]
        context = item.get("context", "")
        
        # Combine context and text (context helps with strategy classification)
        if context:
            input_text = f"{context} </s></s> {text}"
        else:
            input_text = text
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Get label
        strategy = item["strategy"]
        label = self.strategy_to_id.get(strategy, 0)  # Default to first strategy if not found
        
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long),
        }


def load_and_prepare_data(
    data_path: str,
    config: TrainingConfig
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, int], List[str]]:
    """
    Load your final_agentic_hr_dataset.json and prepare it for training.
    
    Returns:
        train_examples, val_examples, test_examples, strategy_to_id, id_to_strategy
    """
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'r') as f:
        sessions = json.load(f)
    
    print(f"Loaded {len(sessions)} negotiation sessions")
    
    # Extract individual turns with context
    examples = []
    
    for session in sessions:
        conversation = session.get("conversation", [])
        context = ""  # Rolling context
        
        for turn in conversation:
            role = turn.get("role", "")
            response = turn.get("response", "")
            strategy = turn.get("negotiation_strategy", "No Strategy")
            
            # Create example
            examples.append({
                "text": response,
                "context": context,
                "strategy": strategy,
                "role": role,
                "scenario_id": session.get("scenario_id", ""),
            })
            
            # Update rolling context (keep last 2 turns for efficiency)
            context = f"{context} {role}: {response}".strip()
            context_turns = context.split(". ")[-4:]  # Keep last ~2 exchanges
            context = ". ".join(context_turns)
    
    print(f"Extracted {len(examples)} individual turns")
    
    # Get unique strategies
    strategies = sorted(list(set(ex["strategy"] for ex in examples)))
    strategy_to_id = {s: i for i, s in enumerate(strategies)}
    id_to_strategy = strategies
    
    print(f"\nFound {len(strategies)} unique negotiation strategies:")
    for s, idx in strategy_to_id.items():
        count = sum(1 for ex in examples if ex["strategy"] == s)
        print(f"  {idx}: {s} ({count} examples, {100*count/len(examples):.1f}%)")
    
    # Split data
    train_val, test = train_test_split(
        examples, 
        test_size=config.test_split, 
        random_state=config.seed,
        stratify=[ex["strategy"] for ex in examples]
    )
    
    train, val = train_test_split(
        train_val,
        test_size=config.val_split / (1 - config.test_split),
        random_state=config.seed,
        stratify=[ex["strategy"] for ex in train_val]
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(train)} examples")
    print(f"  Val:   {len(val)} examples")
    print(f"  Test:  {len(test)} examples")
    
    return train, val, test, strategy_to_id, id_to_strategy


def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    device,
    max_grad_norm
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        logits = outputs.logits
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=-1)
        predictions.extend(preds.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())
        
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    return avg_loss, accuracy, f1_macro


def evaluate(model, dataloader, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item()
            
            preds = torch.argmax(logits, dim=-1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(true_labels, predictions)
    f1_macro = f1_score(true_labels, predictions, average='macro')
    
    return avg_loss, accuracy, f1_macro, predictions, true_labels


def main(args):
    # Configuration
    config = TrainingConfig(
        save_dir=args.save_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    
    print("="*60)
    print("GcNS Classifier Training")
    print("="*60)
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Save directory: {config.save_dir}")
    print("="*60)
    
    # Load and prepare data
    train_data, val_data, test_data, strategy_to_id, id_to_strategy = load_and_prepare_data(
        args.data_path,
        config
    )
    
    # Save strategy mapping
    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'strategy_mapping.json'), 'w') as f:
        json.dump({
            'strategy_to_id': strategy_to_id,
            'id_to_strategy': id_to_strategy
        }, f, indent=2)
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = RobertaTokenizer.from_pretrained(config.model_name)
    model = RobertaForSequenceClassification.from_pretrained(
        config.model_name,
        num_labels=len(strategy_to_id)
    )
    model.to(config.device)
    
    # Create datasets
    train_dataset = NegotiationStrategyDataset(train_data, tokenizer, strategy_to_id, config.max_length)
    val_dataset = NegotiationStrategyDataset(val_data, tokenizer, strategy_to_id, config.max_length)
    test_dataset = NegotiationStrategyDataset(test_data, tokenizer, strategy_to_id, config.max_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    
    # Setup optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0
    
    for epoch in range(config.num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{config.num_epochs}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, optimizer, scheduler, config.device, config.max_grad_norm
        )
        
        print(f"\nTraining   - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        
        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, config.device)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"New best validation F1: {val_f1:.4f}! Saving model...")
            
            model.save_pretrained(config.save_dir)
            tokenizer.save_pretrained(config.save_dir)
            
            # Save training info
            with open(os.path.join(config.save_dir, 'training_info.json'), 'w') as f:
                json.dump({
                    'best_epoch': epoch + 1,
                    'val_accuracy': val_acc,
                    'val_f1_macro': val_f1,
                    'num_strategies': len(strategy_to_id),
                    'config': {
                        'model_name': config.model_name,
                        'max_length': config.max_length,
                        'batch_size': config.batch_size,
                        'learning_rate': config.learning_rate,
                    }
                }, f, indent=2)
    
    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Final Evaluation on Test Set")
    print(f"{'='*60}")
    
    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, config.device
    )
    
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Macro F1: {test_f1:.4f} ({test_f1*100:.2f}%)")
    print(f"  Loss: {test_loss:.4f}")
    
    # Classification report
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        test_labels, 
        test_preds, 
        target_names=id_to_strategy,
        digits=3
    ))
    
    # Save test results
    with open(os.path.join(config.save_dir, 'test_results.json'), 'w') as f:
        json.dump({
            'test_accuracy': test_acc,
            'test_f1_macro': test_f1,
            'test_loss': test_loss,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best model saved to: {config.save_dir}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Test F1: {test_f1:.4f}")
    print(f"{'='*60}")
    
    return config.save_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GcNS Negotiation Strategy Classifier")
    parser.add_argument(
        "--data_path",
        type=str,
        default="final_agentic_hr_dataset.json",
        help="Path to your dataset JSON file"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="trained_models/gcns_negotiation_classifier",
        help="Directory to save trained model"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=4,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Training batch size"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    
    args = parser.parse_args()
    
    model_path = main(args)
    
    print("\n" + "="*60)
    print("TO USE THIS MODEL IN YOUR PIPELINE:")
    print("="*60)
    print(f"In strategy_classifier.py, use:")
    print(f"  model_path = '{model_path}'")
    print("="*60)