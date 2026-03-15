class DatasetConfig:
    def __init__(self, path, max_art_len=400, max_sum_len=50, split=0.8):
        self.path = path
        self.max_art_len = max_art_len
        self.max_sum_len = max_sum_len
        self.split = split


class TokenConfig:
    def __init__(self, min_freq=1):
        self.min_freq = min_freq


class ModelConfig:
    def __init__(self, name, emb=128, hid=256, layers=1, heads=4, ff=256):
        if hasattr(name, "value"):
            name = name.value
        self.name = name
        self.emb = emb
        self.hid = hid
        self.layers = layers
        self.heads = heads
        self.ff = ff


class TrainConfig:
    def __init__(self, bs=8, lr=0.001, epoch=1, cov=1.0, tf=0.5, seed=7):
        self.bs = bs
        self.lr = lr
        self.epoch = epoch
        self.cov = cov
        self.tf = tf
        self.seed = seed
