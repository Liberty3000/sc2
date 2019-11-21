import numpy as np
from sc2.policy import Policy

class Greedy(Policy):
    def __init__(self, ):
        super().__init__()
    def __call__(self, logits, mask=None):
        if not mask is None:
            logits = logits[:,mask.long()]
        return logits.argmax().item()
