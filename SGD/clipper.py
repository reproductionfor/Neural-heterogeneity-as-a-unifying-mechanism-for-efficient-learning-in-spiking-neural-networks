import torch
import numpy as np


class ZeroOneClipper(object):
    def __call__(self, module):
     
        if hasattr(module, 'alpha'):
          
            module.alpha.data.clamp_(2/np.e, 0.995)
        if hasattr(module, 'beta'):
            module.beta.data.clamp_(2/np.e, 0.995)
        if hasattr(module, 'th'):
            module.th.data.clamp_(0.5, 1.5)
