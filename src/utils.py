import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.sum=0; self.count=0
    def update(self, val, n=1): self.sum += val*n; self.count += n
    @property
    def avg(self): return self.sum / max(1,self.count)
