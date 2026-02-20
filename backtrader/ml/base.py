#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from abc import abstractmethod

from ..metabase import MetaParams
from ..utils.py3 import with_metaclass

from .state import ModelState
from .exceptions import NotFittedError


class BaseMLModel(with_metaclass(MetaParams, object)):
    '''
    Abstract base class for all ML models in backtrader.
    
    This class defines the contract that any ML model must implement
    to integrate with the backtrader framework. It is framework-agnostic
    and can be subclassed for neural networks, sklearn models, etc.
    
    Subclasses must implement:
        - fit(X, y, **kwargs): Train the model
        - predict(X): Generate predictions
        - save(path): Serialize model to disk
        - load(path): Load model from disk
        - _get_n_parameters(): Return number of trainable parameters
    '''
    
    params = (
        ('input_dim', None),
        ('output_dim', 1),
        ('seed', None),
    )
    
    def __init__(self):
        self._state = ModelState.UNTRAINED
        self._training_metadata = {}
        
        if self.p.seed is not None:
            self._set_seed(self.p.seed)
    
    @property
    def state(self):
        '''Returns the current ModelState.'''
        return self._state
    
    @property
    def n_parameters(self):
        '''Returns the number of trainable parameters.'''
        return self._get_n_parameters()
    
    @property
    def training_metadata(self):
        '''Returns metadata from the last training run.'''
        return self._training_metadata.copy()
    
    def is_fitted(self):
        '''Returns True if the model has been trained.'''
        return self._state == ModelState.TRAINED
    
    def reset(self):
        '''
        Reset the model to its untrained state.
        
        Subclasses should override this to also reset weights/parameters.
        '''
        self._state = ModelState.UNTRAINED
        self._training_metadata = {}
    
    def _set_seed(self, seed):
        '''
        Set random seed for reproducibility.
        
        Subclasses should override to set framework-specific seeds.
        '''
        pass
    
    @abstractmethod
    def fit(self, X, y, **kwargs):
        '''
        Train the model on the provided data.
        
        Args:
            X: Training features (array-like)
            y: Training targets (array-like)
            **kwargs: Additional training parameters
            
        Returns:
            self
        '''
        raise NotImplementedError
    
    @abstractmethod
    def predict(self, X):
        '''
        Generate predictions for the provided features.
        
        Args:
            X: Features to predict on (array-like)
            
        Returns:
            Predictions as numpy array
            
        Raises:
            NotFittedError: If the model has not been fitted
        '''
        raise NotImplementedError
    
    @abstractmethod
    def save(self, path):
        '''
        Save the model to disk.
        
        Args:
            path: File path to save to
        '''
        raise NotImplementedError
    
    @abstractmethod
    def load(self, path):
        '''
        Load the model from disk.
        
        Args:
            path: File path to load from
            
        Returns:
            self
        '''
        raise NotImplementedError
    
    @abstractmethod
    def _get_n_parameters(self):
        '''
        Return the number of trainable parameters.
        
        Returns:
            int: Number of trainable parameters
        '''
        raise NotImplementedError
    
    def _check_fitted(self):
        '''Raise NotFittedError if model is not fitted.'''
        if not self.is_fitted():
            raise NotFittedError(
                f'{self.__class__.__name__} is not fitted. '
                'Call fit() before predict().'
            )
    
    def __repr__(self):
        params_str = ', '.join(
            f'{k}={v}' for k, v in self.p._getkwargs().items()
        )
        return f'{self.__class__.__name__}({params_str})'
