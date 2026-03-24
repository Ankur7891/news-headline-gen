from core.base import BaseModel
from wrappers import engine as en


class Attn(en.Module):
    def __init__(self, eh, dh):
        super().__init__()
        self.w1 = en.Linear(eh, dh)
        self.w2 = en.Linear(dh, dh)
        self.w3 = en.Linear(1, dh)
        self.v = en.Linear(dh, 1)

    def forward(self, enc, dec, cov):
        src = enc.size(1)
        d = dec.unsqueeze(1).expand(-1, src, -1)
        c = cov.unsqueeze(2)
        s = self.v(en.tanh(self.w1(enc) + self.w2(d) + self.w3(c)))
        s = s.squeeze(2)
        a = en.softmax(s, dim=1)
        ctx = en.bmm(a.unsqueeze(1), enc).squeeze(1)
        cov = cov + a
        return ctx, a, cov


class LSTMEnc(en.Module):
    def __init__(self, v, e, h):
        super().__init__()
        self.emb = en.Embedding(v, e, padding_idx=0)
        self.lstm = en.LSTM(e, h, num_layers=1, batch_first=True, bidirectional=True)
        self.rh = en.Linear(h * 2, h)
        self.rc = en.Linear(h * 2, h)

    def forward(self, x):
        e = self.emb(x)
        out, (h, c) = self.lstm(e)
        hcat = en.cat([h[0], h[1]], dim=1)
        ccat = en.cat([c[0], c[1]], dim=1)
        h0 = en.tanh(self.rh(hcat)).unsqueeze(0)
        c0 = en.tanh(self.rc(ccat)).unsqueeze(0)
        return out, (h0, c0)


class LSTMDec(en.Module):
    def __init__(self, v, e, eh, dh):
        super().__init__()
        self.emb = en.Embedding(v, e, padding_idx=0)
        self.lstm = en.LSTM(e, dh, num_layers=1, batch_first=True)
        self.attn = Attn(eh, dh)
        self.pg = en.Linear(eh + dh + e, 1)
        self.vp = en.Linear(eh + dh, v)

    def forward(self, x, h, enc, cov, enc_ids):
        e = self.emb(x)
        out, h = self.lstm(e, h)
        d = out.squeeze(1)
        ctx, a, cov = self.attn(enc, d, cov)
        cat = en.cat([d, ctx], dim=1)
        vd = en.softmax(self.vp(cat), dim=1)
        pin = en.cat([ctx, d, e.squeeze(1)], dim=1)
        p = en.sigmoid(self.pg(pin))
        fin = p * vd
        if enc_ids is not None:
            cp = en.zeros_like(vd)
            cp.scatter_add_(1, enc_ids, a)
            fin = fin + (1 - p) * cp
        return fin, h, cov, a


class LSTMSeq(BaseModel, en.Module):
    def __init__(self, cfg, vocab):
        en.Module.__init__(self)
        self.cfg = cfg
        self.vocab = vocab
        self.enc = LSTMEnc(vocab, cfg.emb, cfg.hid)
        self.dec = LSTMDec(vocab, cfg.emb, cfg.hid * 2, cfg.hid)

    def forward(self, art, summ):
        enc, h = self.enc(art)
        b = art.size(0)
        src = art.size(1)
        cov = en.zeros((b, src), device=art.device)
        logp = []
        cov_loss = 0.0
        t = 0
        while t < summ.size(1) - 1:
            x = summ[:, t].unsqueeze(1)
            prev = cov
            dist, h, cov, a = self.dec(x, h, enc, cov, art)
            dist = en.clip(dist, 1e-9, 1.0)
            lp = en.log(dist)
            logp.append(lp.unsqueeze(1))
            m = en.minv(a, prev)
            cov_loss = cov_loss + en.sumv(m)
            t += 1
        out = en.cat(logp, dim=1)
        cov_loss = cov_loss / float(b)
        return out, cov_loss

    def generate(self, art, max_len):
        enc, h = self.enc(art)
        b = art.size(0)
        src = art.size(1)
        cov = en.zeros((b, src), device=art.device)
        out = []
        x = en.zeros((b, 1), dtype=art.dtype, device=art.device)
        x = x + 1
        t = 0
        while t < max_len:
            dist, h, cov, a = self.dec(x, h, enc, cov, art)
            y = en.argmax(dist, dim=1)
            if y.item() == 2:
                break
            out.append(y.unsqueeze(1))
            x = y.unsqueeze(1)
            t += 1
        if len(out) == 0:
            return en.zeros((b, 1), dtype=art.dtype, device=art.device)
        return en.cat(out, dim=1)
