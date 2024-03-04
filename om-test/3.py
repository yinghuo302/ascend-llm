import numpy as np
import time
from engine import ACLModel,initResource
ctx = initResource(0)
session = ACLModel('/root/program/mlp.om',ctx)
input_data = np.random.randn(1, 1000).astype(np.float32)
print(session.inference([input_data]))