from core.base import BaseTok
from utils import text as tx


class WordTok(BaseTok):
    def __init__(self, cfg):
        self.cfg = cfg
        self.w2i = {}
        self.i2w = {}
        self.w2i["<PAD>"] = 0
        self.w2i["<SOS>"] = 1
        self.w2i["<EOS>"] = 2
        self.w2i["<UNK>"] = 3
        self.i2w[0] = "<PAD>"
        self.i2w[1] = "<SOS>"
        self.i2w[2] = "<EOS>"
        self.i2w[3] = "<UNK>"

    def build_vocab(self, xs):
        cnt = {}
        i = 0
        while i < len(xs):
            parts = tx.split(xs[i])
            j = 0
            while j < len(parts):
                w = parts[j]
                if w in cnt:
                    cnt[w] = cnt[w] + 1
                else:
                    cnt[w] = 1
                j += 1
            i += 1
        idx = 4
        for w in cnt:
            if cnt[w] >= self.cfg.min_freq:
                if w not in self.w2i:
                    self.w2i[w] = idx
                    self.i2w[idx] = w
                    idx += 1

    def encode(self, s, max_len):
        parts = tx.split(s)
        ids = []
        ids.append(1)
        i = 0
        while i < len(parts):
            w = parts[i]
            if w in self.w2i:
                ids.append(self.w2i[w])
            else:
                ids.append(3)
            i += 1
        ids.append(2)
        if len(ids) < max_len:
            pad = max_len - len(ids)
            j = 0
            while j < pad:
                ids.append(0)
                j += 1
        else:
            ids = ids[:max_len]
        return ids

    def decode(self, ids):
        out = []
        i = 0
        while i < len(ids):
            k = ids[i]
            if k == 0:
                i += 1
                continue
            if k == 2:
                break
            if k in self.i2w:
                out.append(self.i2w[k])
            else:
                out.append("<UNK>")
            i += 1
        return tx.join(out)

    def size(self):
        return len(self.w2i)
