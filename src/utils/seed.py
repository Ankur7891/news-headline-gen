import random
from wrappers import engine as en


def set_seed(n):
    random.seed(n)
    en.seed(n)
