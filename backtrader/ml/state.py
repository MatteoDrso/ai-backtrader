#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

from enum import Enum


class ModelState(Enum):
    '''
    Represents the current state of an ML model.
    '''
    UNTRAINED = 'untrained'
    TRAINING = 'training'
    TRAINED = 'trained'
    FAILED = 'failed'
