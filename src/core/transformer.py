import math
from core.base import BaseModel
from wrappers import engine as en


class MHA(en.Module):
    def __init__(self, d, h):
        super().__init__()
        self.d = d
        self.h = h
        self.dh = int(d / h)
        self.wq = en.Linear(d, d)
        self.wk = en.Linear(d, d)
        self.wv = en.Linear(d, d)
        self.wo = en.Linear(d, d)

    def split(self, x):
        b = x.size(0)
        t = x.size(1)
        x = x.view(b, t, self.h, self.dh)
        x = en.transpose(x, 1, 2)
        return x

    def merge(self, x):
        b = x.size(0)
        t = x.size(2)
        x = en.transpose(x, 1, 2)
        x = x.contiguous().view(b, t, self.d)
        return x

    def forward(self, q, kv=None, mask=None):
        if kv is None:
            kv = q
        qq = self.wq(q)
        kk = self.wk(kv)
        vv = self.wv(kv)
        qq = self.split(qq)
        kk = self.split(kk)
        vv = self.split(vv)
        sc = 1.0 / math.sqrt(self.dh)
        score = en.matmul(qq, en.transpose(kk, -2, -1)) * sc
        if mask is not None:
            score = score + mask
        attn = en.softmax(score, dim=-1)
        ctx = en.matmul(attn, vv)
        ctx = self.merge(ctx)
        out = self.wo(ctx)
        return out, attn


class FF(en.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.l1 = en.Linear(d, ff)
        self.l2 = en.Linear(ff, d)

    def forward(self, x):
        x = self.l1(x)
        x = en.relu(x)
        x = self.l2(x)
        return x


class EncLayer(en.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.mha = MHA(d, h)
        self.n1 = en.LayerNorm(d)
        self.ff = FF(d, ff)
        self.n2 = en.LayerNorm(d)

    def forward(self, x, mask):
        a, attn = self.mha(x, None, mask)
        x = self.n1(x + a)
        f = self.ff(x)
        x = self.n2(x + f)
        return x, attn


class DecLayer(en.Module):
    def __init__(self, d, h, ff):
        super().__init__()
        self.m1 = MHA(d, h)
        self.n1 = en.LayerNorm(d)
        self.m2 = MHA(d, h)
        self.n2 = en.LayerNorm(d)
        self.ff = FF(d, ff)
        self.n3 = en.LayerNorm(d)

    def forward(self, x, enc, sm, tm):
        a1, att1 = self.m1(x, None, tm)
        x = self.n1(x + a1)
        a2, att2 = self.m2(x, enc, sm)
        x = self.n2(x + a2)
        f = self.ff(x)
        x = self.n3(x + f)
        return x, att1, att2


class TransSeq(BaseModel, en.Module):
    def __init__(self, cfg, vocab):
        en.Module.__init__(self)
        self.cfg = cfg
        self.vocab = vocab
        self.emb = en.Embedding(vocab, cfg.emb, padding_idx=0)
        self.pos = en.Embedding(512, cfg.emb)
        el = []
        dl = []
        i = 0
        while i < cfg.layers:
            el.append(EncLayer(cfg.emb, cfg.heads, cfg.ff))
            dl.append(DecLayer(cfg.emb, cfg.heads, cfg.ff))
            i += 1
        self.enc = en.ModuleList(el)
        self.dec = en.ModuleList(dl)
        self.out = en.Linear(cfg.emb, vocab)

    def src_mask(self, src):
        m = (src != 0).unsqueeze(1).unsqueeze(2)
        m = m.float()
        m = (1.0 - m) * -1e9
        return m

    def tgt_mask(self, tgt):
        t = tgt.size(1)
        a = en.ones((t, t), device=tgt.device)
        a = en.tril(a)
        a = a.unsqueeze(0).unsqueeze(1)
        b = (tgt != 0).unsqueeze(1).unsqueeze(2)
        b = b.float()
        m = a * b
        m = (1.0 - m) * -1e9
        return m

    def encode(self, src):
        b = src.size(0)
        t = src.size(1)
        pos = en.arange(t, device=src.device).unsqueeze(0).expand(b, t)
        x = self.emb(src) + self.pos(pos)
        sm = self.src_mask(src)
        i = 0
        while i < len(self.enc):
            x, attn = self.enc[i](x, sm)
            i += 1
        return x, sm

    def decode(self, tgt, enc, sm):
        b = tgt.size(0)
        t = tgt.size(1)
        pos = en.arange(t, device=tgt.device).unsqueeze(0).expand(b, t)
        x = self.emb(tgt) + self.pos(pos)
        tm = self.tgt_mask(tgt)
        i = 0
        while i < len(self.dec):
            x, a1, a2 = self.dec[i](x, enc, sm, tm)
            i += 1
        out = self.out(x)
        logp = en.log_softmax(out, dim=-1)
        return logp

    def forward(self, art, summ):
        enc, sm = self.encode(art)
        logp = self.decode(summ[:, :-1], enc, sm)
        cov_loss = 0.0
        return logp, cov_loss

    def generate(self, art, max_len):
        enc, sm = self.encode(art)
        b = art.size(0)
        ys = en.zeros((b, 1), dtype=art.dtype, device=art.device)
        ys = ys + 1
        t = 0
        while t < max_len:
            logp = self.decode(ys, enc, sm)
            last = logp[:, -1, :]
            y = en.argmax(last, dim=1)
            ys = en.cat([ys, y.unsqueeze(1)], dim=1)
            t += 1
        return ys[:, 1:]
