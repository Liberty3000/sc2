from sc2.policy import Policy

class AlwaysZero(Policy):
    def __init__(self):
        super().__init__()
    def __call__(self, *args, **kwargs):
        return 0
