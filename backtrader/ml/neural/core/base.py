#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import abstractmethod
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from ...base import BaseMLModel
from ...state import ModelState
from ...exceptions import (
    ModelBuildError,
    TrainingError,
    InvalidParameterError,
    NotFittedError,
)


class NeuralNetworkBase(BaseMLModel):
    '''
    Abstract base class for PyTorch neural networks in backtrader.
    
    This class provides the infrastructure for training and inference
    of neural networks. Subclasses must implement _build_module() to
    define the network architecture.
    
    The model uses composition rather than inheritance from nn.Module
    to avoid metaclass conflicts with backtrader's MetaParams system.
    
    Subclasses must implement:
        - _build_module(): Construct and return the nn.Module
    
    Example:
        class MyNetwork(NeuralNetworkBase):
            params = (
                ('hidden_units', 64),
            )
            
            def _build_module(self):
                return nn.Sequential(
                    nn.Linear(self.p.input_dim, self.p.hidden_units),
                    nn.ReLU(),
                    nn.Linear(self.p.hidden_units, self.p.output_dim),
                )
    '''
    
    params = (
        ('epochs', 100),
        ('batch_size', 32),
        ('learning_rate', 0.001),
        ('optimizer', 'adam'),
        ('weight_decay', 0.0),
        ('loss_fn', 'mse'),
        ('device', 'auto'),
        ('dropout', 0.0),
        ('verbose', True),
        ('validation_split', 0.0),
    )
    
    _OPTIMIZERS = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop,
    }
    
    _LOSS_FUNCTIONS = {
        'mse': nn.MSELoss,
        'mae': nn.L1Loss,
        'huber': nn.SmoothL1Loss,
        'cross_entropy': nn.CrossEntropyLoss,
        'bce': nn.BCELoss,
        'bce_logits': nn.BCEWithLogitsLoss,
    }
    
    def __init__(self):
        super(NeuralNetworkBase, self).__init__()
        
        self._device = self._resolve_device()
        self._module = None
        self._optimizer = None
        self._loss_fn_instance = None
        self._training_history = {'train_loss': [], 'val_loss': []}
        
        self._build_and_initialize()
    
    def _build_and_initialize(self):
        '''Build the module and move to device.'''
        if self.p.input_dim is None:
            return
        
        try:
            self._module = self._build_module()
            if self._module is not None:
                self._module = self._module.to(self._device)
                self._optimizer = self._create_optimizer()
                self._loss_fn_instance = self._create_loss_fn()
        except Exception as e:
            raise ModelBuildError(f'Failed to build module: {e}')
    
    @abstractmethod
    def _build_module(self):
        '''
        Construct and return the PyTorch nn.Module.
        
        This method is called during __init__ after params are set.
        Use self.p.input_dim, self.p.output_dim, etc. to configure
        the architecture.
        
        Returns:
            nn.Module: The constructed neural network
        '''
        raise NotImplementedError
    
    def _resolve_device(self):
        '''Resolve the compute device.'''
        if self.p.device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(self.p.device)
    
    def _create_optimizer(self):
        '''Create the optimizer instance.'''
        opt_name = self.p.optimizer.lower()
        if opt_name not in self._OPTIMIZERS:
            raise InvalidParameterError(
                f"Unknown optimizer '{self.p.optimizer}'. "
                f"Available: {list(self._OPTIMIZERS.keys())}"
            )
        
        opt_class = self._OPTIMIZERS[opt_name]
        opt_kwargs = {
            'lr': self.p.learning_rate,
            'weight_decay': self.p.weight_decay,
        }
        
        if opt_name == 'sgd':
            opt_kwargs.pop('weight_decay')
        
        return opt_class(self._module.parameters(), **opt_kwargs)
    
    def _create_loss_fn(self):
        '''Create the loss function instance.'''
        loss_name = self.p.loss_fn.lower()
        if loss_name not in self._LOSS_FUNCTIONS:
            raise InvalidParameterError(
                f"Unknown loss function '{self.p.loss_fn}'. "
                f"Available: {list(self._LOSS_FUNCTIONS.keys())}"
            )
        
        return self._LOSS_FUNCTIONS[loss_name]()
    
    def _set_seed(self, seed):
        '''Set random seeds for reproducibility.'''
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
    
    def _to_tensor(self, X, dtype=torch.float32):
        '''Convert input to tensor on device.'''
        if isinstance(X, torch.Tensor):
            return X.to(self._device, dtype=dtype)
        return torch.tensor(X, dtype=dtype, device=self._device)
    
    def _to_numpy(self, tensor):
        '''Convert tensor to numpy array.'''
        return tensor.detach().cpu().numpy()
    
    def _prepare_data(self, X, y):
        '''Prepare data tensors.'''
        X_tensor = self._to_tensor(X)
        y_tensor = self._to_tensor(y)
        
        if y_tensor.ndim == 1:
            y_tensor = y_tensor.unsqueeze(1)
        
        return X_tensor, y_tensor
    
    def _split_validation(self, X, y, validation_split):
        '''Split data into training and validation sets.'''
        n_samples = X.shape[0]
        n_val = int(n_samples * validation_split)
        
        if n_val == 0:
            return X, y, None, None
        
        X_train = X[:-n_val]
        y_train = y[:-n_val]
        X_val = X[-n_val:]
        y_val = y[-n_val:]
        
        return X_train, y_train, X_val, y_val
    
    def fit(self, X, y, validation_data=None, **kwargs):
        '''
        Train the neural network.
        
        Args:
            X: Training features (array-like of shape [n_samples, n_features])
            y: Training targets (array-like of shape [n_samples] or [n_samples, n_outputs])
            validation_data: Optional tuple (X_val, y_val) for validation.
                            If None and validation_split > 0, will split from training data.
            **kwargs: Additional arguments (unused, for API compatibility)
            
        Returns:
            self
        '''
        if self._module is None:
            if self.p.input_dim is None:
                self.params.input_dim = X.shape[1] if hasattr(X, 'shape') else len(X[0])
                self._build_and_initialize()
            else:
                raise ModelBuildError('Module not built. Check input_dim parameter.')
        
        self._state = ModelState.TRAINING
        self._training_history = {'train_loss': [], 'val_loss': []}
        
        try:
            X_tensor, y_tensor = self._prepare_data(X, y)
            
            X_val, y_val = None, None
            if validation_data is not None:
                X_val, y_val = self._prepare_data(*validation_data)
            elif self.p.validation_split > 0:
                X_tensor, y_tensor, X_val, y_val = self._split_validation(
                    X_tensor, y_tensor, self.p.validation_split
                )
            
            dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                dataset,
                batch_size=self.p.batch_size,
                shuffle=False,
            )
            
            self._training_loop(dataloader, X_val, y_val)
            
            self._state = ModelState.TRAINED
            self._training_metadata['epochs_trained'] = self.p.epochs
            self._training_metadata['final_train_loss'] = self._training_history['train_loss'][-1]
            if self._training_history['val_loss']:
                self._training_metadata['final_val_loss'] = self._training_history['val_loss'][-1]
            
        except Exception as e:
            self._state = ModelState.FAILED
            raise TrainingError(f'Training failed: {e}')
        
        return self
    
    def _training_loop(self, dataloader, X_val, y_val):
        '''Execute the training loop.'''
        self._module.train()
        
        for epoch in range(self.p.epochs):
            epoch_loss = 0.0
            n_batches = 0
            
            for X_batch, y_batch in dataloader:
                self._optimizer.zero_grad()
                
                output = self._module(X_batch)
                loss = self._loss_fn_instance(output, y_batch)
                
                loss.backward()
                self._optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            avg_train_loss = epoch_loss / n_batches
            self._training_history['train_loss'].append(avg_train_loss)
            
            val_loss = None
            if X_val is not None:
                val_loss = self._compute_validation_loss(X_val, y_val)
                self._training_history['val_loss'].append(val_loss)
            
            if self.p.verbose:
                self._print_epoch(epoch, avg_train_loss, val_loss)
    
    def _compute_validation_loss(self, X_val, y_val):
        '''Compute loss on validation set.'''
        self._module.eval()
        with torch.no_grad():
            output = self._module(X_val)
            loss = self._loss_fn_instance(output, y_val)
        self._module.train()
        return loss.item()
    
    def _print_epoch(self, epoch, train_loss, val_loss):
        '''Print epoch progress.'''
        msg = f'Epoch {epoch + 1}/{self.p.epochs} - loss: {train_loss:.6f}'
        if val_loss is not None:
            msg += f' - val_loss: {val_loss:.6f}'
        print(msg)
    
    def predict(self, X):
        '''
        Generate predictions.
        
        Args:
            X: Features to predict on (array-like)
            
        Returns:
            numpy.ndarray: Predictions
        '''
        self._check_fitted()
        
        X_tensor = self._to_tensor(X)
        
        self._module.eval()
        with torch.no_grad():
            output = self._module(X_tensor)
        
        return self._to_numpy(output)
    
    def forward(self, x):
        '''
        Raw forward pass (tensor in, tensor out).
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        '''
        if self._module is None:
            raise ModelBuildError('Module not built.')
        return self._module(x)
    
    def save(self, path):
        '''
        Save model to disk.
        
        Args:
            path: File path to save to
        '''
        state = {
            'module_state_dict': self._module.state_dict() if self._module else None,
            'optimizer_state_dict': self._optimizer.state_dict() if self._optimizer else None,
            'training_history': self._training_history,
            'training_metadata': self._training_metadata,
            'model_state': self._state.value,
            'params': self.p._getkwargs(),
        }
        torch.save(state, path)
    
    def load(self, path):
        '''
        Load model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            self
        '''
        state = torch.load(path, map_location=self._device)
        
        if self._module is None:
            for key, value in state['params'].items():
                if hasattr(self.p, key):
                    setattr(self.p, key, value)
            self._build_and_initialize()
        
        if state['module_state_dict'] and self._module:
            self._module.load_state_dict(state['module_state_dict'])
        
        if state['optimizer_state_dict'] and self._optimizer:
            self._optimizer.load_state_dict(state['optimizer_state_dict'])
        
        self._training_history = state.get('training_history', {'train_loss': [], 'val_loss': []})
        self._training_metadata = state.get('training_metadata', {})
        self._state = ModelState(state.get('model_state', 'untrained'))
        
        return self
    
    def reset(self):
        '''Reset the model to untrained state with fresh weights.'''
        super(NeuralNetworkBase, self).reset()
        self._training_history = {'train_loss': [], 'val_loss': []}
        self._build_and_initialize()
    
    def _get_n_parameters(self):
        '''Return number of trainable parameters.'''
        if self._module is None:
            return 0
        return sum(p.numel() for p in self._module.parameters() if p.requires_grad)
    
    def to(self, device):
        '''
        Move model to specified device.
        
        Args:
            device: Device string ('cpu', 'cuda', 'cuda:0', etc.)
            
        Returns:
            self
        '''
        self._device = torch.device(device)
        if self._module is not None:
            self._module = self._module.to(self._device)
        return self
    
    def train_mode(self):
        '''Set module to training mode.'''
        if self._module is not None:
            self._module.train()
        return self
    
    def eval_mode(self):
        '''Set module to evaluation mode.'''
        if self._module is not None:
            self._module.eval()
        return self
    
    def parameters(self):
        '''Return model parameters (proxy to module.parameters()).'''
        if self._module is None:
            return iter([])
        return self._module.parameters()
    
    def state_dict(self):
        '''Return module state dict.'''
        if self._module is None:
            return {}
        return self._module.state_dict()
    
    def load_state_dict(self, state_dict):
        '''Load module state dict.'''
        if self._module is not None:
            self._module.load_state_dict(state_dict)
        return self
    
    @property
    def device(self):
        '''Return current device.'''
        return self._device
    
    @property
    def module(self):
        '''Return the underlying nn.Module.'''
        return self._module
    
    @property
    def training_history(self):
        '''Return training history dict.'''
        return self._training_history.copy()