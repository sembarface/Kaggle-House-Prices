import copy

import numpy as np
import pandas as pd

from tqdm import tqdm

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from torch.amp.grad_scaler import GradScaler

from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR


class DNN(nn.Module):
    """Implement a neural network for regression."""
    def __init__(self, output_size=1, epochs=100, device='cpu', batch_size=32, 
                learning_rate=3e-4, optimizer_name='Adam'):
        """Initialize the neural network and training parameters."""
        
        super().__init__()
        
        self.output_size = output_size
        self.epochs = epochs
        self.device = device
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        
        
    def _build_model(self, input_size):
        """Build model for the custom input size."""
        
        self.mlp = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, self.output_size),
        )
        
        
    def forward(self, x):
        """Perform the forward pass through the network."""
        out = self.mlp(x)
        return out
    
    def fit(self, X_train: pd.DataFrame, y_train: pd.DataFrame, 
            X_val: pd.DataFrame, y_val: pd.DataFrame, verbose=False):
        """Train the network and restore the best validation weights."""
        

        # Prepare data
        X_train_tensor = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val.to_numpy(), dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1,1)
        y_val_tensor = torch.tensor(y_val.to_numpy(), dtype=torch.float32).reshape(-1,1)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, drop_last=True)
        
        
        self.input_size = X_train_tensor.shape[1]
        self._build_model(self.input_size)
        self.to(self.device)
        

        loss_function = torch.nn.MSELoss()
        optimizer = get_optimizer(self.optimizer_name, params=self.parameters(), lr=self.learning_rate)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs, eta_min=3e-7)
        
        scaler = GradScaler(self.device)
        
        
        self.val_losses_ = []
        self.train_losses_ = []
        self.best_val_loss_ = torch.inf
        best_state = None
        
        # Training
        for epoch in tqdm(range(self.epochs)):
            
            self.train()
            train_loss = 0.0
            # Train loop
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                optimizer.zero_grad()
                
                with torch.autocast(self.device):
                    outputs = self(X_batch)
                    loss = loss_function(outputs, y_batch)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                
            train_loss /= len(train_loader)
            self.train_losses_.append(train_loss)
            
            if verbose:
                print(f"Training Loss: {train_loss:.4f}")
            
            
            self.eval()
            val_loss = 0.0
            # Validation loop
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                    with torch.autocast(self.device):
                        outputs = self(X_batch)
                        loss = loss_function(outputs, y_batch)
                    
                    val_loss += loss.item()
                
            val_loss /= len(val_loader)
            self.val_losses_.append(val_loss)
            
            # Remember best_state
            if val_loss < self.best_val_loss_:
                self.best_epoch_ = epoch + 1
                self.best_val_loss_ = val_loss
                best_state = copy.deepcopy(self.state_dict())
            
            if verbose:
                print((f"Validation Loss: {val_loss:.4f}"))
            
            scheduler.step()
        
        if verbose:
            print(f'Best epoch: {self.best_epoch_}')
            plt.plot(self.train_losses_,label='train')
            plt.plot(self.val_losses_, label='val')
            plt.legend()
            plt.show()

        self.load_state_dict(best_state)
        return self
    
    def predict(self, X: pd.DataFrame):
        """Predict class probabilities for input samples."""
        
        self.eval()
        with torch.no_grad():
            
            X_tensor = torch.tensor(X.to_numpy(), dtype=torch.float32).to(self.device)
            return self(X_tensor).cpu().numpy().ravel()
    
    
    def get_checkpoint(self):
        """Create a checkpoint containing model weights and training history."""
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'train_losses': self.train_losses_,
            'val_losses' : self.val_losses_,
            'best_epoch': self.best_epoch_
        } 
        return checkpoint
    
    def load_checkpoint(self, checkpoint):
        """Restore the model from a saved checkpoint."""
        
        self.input_size = checkpoint['input_size']
        self._build_model(self.input_size)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.to(self.device)
        
        self.train_losses_ = checkpoint['train_losses']
        self.val_losses_ = checkpoint['val_losses']
        self.best_epoch_ = checkpoint['best_epoch']


def get_optimizer(optimizer_name, params, lr):
    """Create an optimizer."""
    
    if optimizer_name == 'Adam':
        optimizer = Adam(params, lr=lr)
    
    elif optimizer_name == 'SGD':
        optimizer = SGD(params, lr=lr)
    
    return optimizer