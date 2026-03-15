from core.base import BaseData
from dtypes.records import Example, Batch
from wrappers import table as tb


class NewsData(BaseData):
    def __init__(self, cfg):
        self.cfg = cfg
        self.arts, self.sums = tb.load_news(cfg.path)
        self.ex = []

    def build(self, tok):
        ex = []
        i = 0
        while i < len(self.arts):
            a = self.arts[i]
            s = self.sums[i]
            aid = tok.encode(a, self.cfg.max_art_len)
            sid = tok.encode(s, self.cfg.max_sum_len)
            ex.append(Example(a, s, aid, sid))
            i += 1
        self.ex = ex
        return ex

    def split(self):
        n = len(self.arts)
        cut = int(self.cfg.split * n)
        tr = []
        te = []
        i = 0
        while i < n:
            if i < cut:
                tr.append(i)
            else:
                te.append(i)
            i += 1
        return tr, te

    def get_batch(self, idxs):
        a = []
        s = []
        i = 0
        while i < len(idxs):
            ex = self.ex[idxs[i]]
            a.append(ex.art_ids)
            s.append(ex.sum_ids)
            i += 1
        return Batch(a, s)

    def vocab_texts(self):
        xs = []
        i = 0
        while i < len(self.arts):
            xs.append(self.arts[i])
            i += 1
        i = 0
        while i < len(self.sums):
            xs.append(self.sums[i])
            i += 1
        return xs
