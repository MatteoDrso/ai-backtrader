#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)


class MLModelError(Exception):
    '''Base exception for all ML model errors.'''
    pass


class NotFittedError(MLModelError):
    '''Raised when predict is called on an unfitted model.'''
    pass


class ModelBuildError(MLModelError):
    '''Raised when model construction fails.'''
    pass


class TrainingError(MLModelError):
    '''Raised when training fails.'''
    pass


class InvalidParameterError(MLModelError):
    '''Raised when an invalid parameter value is provided.'''
    pass
