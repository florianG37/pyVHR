import numpy as np
from .base import VHRMethod

class MAP_3DCNN(VHRMethod):
    methodName = 'MAP_3DCNN'
    
    def __init__(self, **kwargs):
        super(MAP_3DCNN, self).__init__(**kwargs)
        
    def apply(self, X):
        bpm = np.asarray([80])
        return bpm