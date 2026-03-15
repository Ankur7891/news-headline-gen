import random
from core.base import BaseTrain
from wrappers import engine as en
from utils import metrics as mt
from utils import seed as sd


class Trainer(BaseTrain):
    def __init__(self, cfg):
        self.cfg = cfg

    def nll(self, logp, tgt):
        idx = tgt.unsqueeze(2)
        sel = en.gather(logp, 2, idx).squeeze(2)
        m = (tgt != 0).float()
        s = -en.sumv(sel * m)
        d = en.sumv(m)
        if float(d) == 0.0:
            return s
        return s / d

    def fit(self, model, data, tok):
        sd.set_seed(self.cfg.seed)
        data.build(tok)
        tr, te = data.split()
        dev = en.device()
        model.to(dev)
        opt = en.adam(model.parameters(), self.cfg.lr)
        e = 0
        while e < self.cfg.epoch:
            random.shuffle(tr)
            model.train()
            i = 0
            while i < len(tr):
                idxs = tr[i:i + self.cfg.bs]
                batch = data.get_batch(idxs)
                art = en.tensor(batch.art_ids, dtype=en.long, device=dev)
                summ = en.tensor(batch.sum_ids, dtype=en.long, device=dev)
                logp, cov = model.forward(art, summ)
                tgt = summ[:, 1:]
                loss = self.nll(logp, tgt)
                loss = loss + self.cfg.cov * cov
                opt.zero_grad()
                loss.backward()
                opt.step()
                i += self.cfg.bs
            self.eval(model, data, tok, te, dev)
            e += 1

    def eval(self, model, data, tok, idxs, dev):
        model.eval()
        if len(idxs) == 0:
            return
        j = 0
        k = 0
        while j < len(idxs) and k < 5:
            b = [idxs[j]]
            batch = data.get_batch(b)
            art = en.tensor(batch.art_ids, dtype=en.long, device=dev)
            pred = model.generate(art, data.cfg.max_sum_len)
            pid = pred[0].tolist()
            out = tok.decode(pid)
            ref = data.ex[idxs[j]].summ
            pa, ra, fa = mt.overlap(out.split(), ref.lower().split())
            j += 1
            k += 1
