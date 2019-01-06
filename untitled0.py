# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:55:04 2019

@author: Meet
"""

### Read squeezenet weights

import scipy.io
import numpy as np

weights_raw = scipy.io.loadmat("sqz_full.mat")

x = weights_raw['conv1']

x_ = x[0][0]
y_ = x[0][1]

x_.astype(np.float32)


