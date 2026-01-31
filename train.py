#Training Script for Multimodal CAG Stent Prediction System

#Last Updated: January 2026

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import argparse
from tqdm import tqdm
import os

from model import create_model
from data_loader import get_data_loaders
from utils import (
    calculate_metrics, 
    save_checkpoint, 
    EarlyStopping,
    set_seed
)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str
) -> float:
    """
    Train model for one epoch
    
    Args:
        model: The neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    progress_bar = tqdm(train_loader, desc="Training")
    
    for batch_idx, (multimodal_data, labels) in enumerate(progress_bar):
        for key in multimodal_data:
            multimodal_data[key] = multimodal_data[key].to(device)
        labels = labels.to(device).squeeze()
        optimizer.zero_grad()
        
        outputs = model(multimodal_data)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        # Update statistics
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> dict:
    """
    Validate model on validation set
    
    Args:
        model: The neural network model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        
    Returns:
        Dictionary containing validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for multimodal_data, labels in tqdm(val_loader, desc="Validation"):
            # Move data to device
            for key in multimodal_data:
                multimodal_data[key] = multimodal_data[key].to(device)
            labels = labels.to(device).squeeze()
            outputs = model(multimodal_data)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    import numpy as np
    metrics = calculate_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs)
    )
    metrics['loss'] = total_loss / len(val_loader)
    
    return metrics


def main(config_path: str):
    """
    Main training function
    
    Args:
        config_path: Path to configuration file
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(42)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Loading datasets...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=config['paths']['data_root'],
        batch_size=config['training']['batch_size'],
        num_workers=config['hardware']['num_workers'],
        modalities=config['dataset']['modalities']
    )
    
    print("Creating model...")
    model_config = {
        'modalities': config['dataset']['modalities'],
        'backbone': config['model']['backbone'],
        'num_heads': config['model']['attention_mechanism']['num_heads'],
        'hidden_dims': config['model']['classifier']['hidden_dims'],
        'num_classes': config['model']['classifier']['output_classes'],
        'dropout': config['model']['classifier']['dropout']
    }
    model = create_model(model_config).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=config['training']['lr_scheduler']['factor'],
        patience=config['training']['lr_scheduler']['patience'],
        min_lr=config['training']['lr_scheduler']['min_lr']
    )
    
    early_stopping = EarlyStopping(
        patience=config['training']['early_stopping']['patience'],
        mode=config['training']['early_stopping']['mode']
    )
    
    print("\nStarting training...")
    best_auc = 0.0
    
    for epoch in range(config['training']['epochs']):
        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")
        print("-" * 50)
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        print(f"Training Loss: {train_loss:.4f}")
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Validation Metrics:")
        for name, value in val_metrics.items():
            print(f"  {name}: {value:.4f}")
        scheduler.step(val_metrics['auc_roc'])
        if val_metrics['auc_roc'] > best_auc:
            best_auc = val_metrics['auc_roc']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                os.path.join(
                    config['paths']['model_checkpoints'],
                    'best_model.pth'
                )
            )
            print(f"New best model saved! AUC: {best_auc:.4f}")
        
        if early_stopping(val_metrics['auc_roc']):
            print("\nEarly stopping triggered!")
            break
    
    print("\nTraining completed!")
    print(f"Best validation AUC: {best_auc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Train Multimodal CAG Stent Prediction Model'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    main(args.config)
