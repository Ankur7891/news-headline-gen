import sys
import random
from core.base import BaseTrain
from wrappers import engine as en
from utils import metrics as mt
from utils import seed as sd
from tqdm import tqdm
from rouge_score import rouge_scorer as rs


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

        total_batches = max(1, len(tr) // self.cfg.bs)

        print(f"\nDevice     : {dev}")
        print(f"Train size : {len(tr)}")
        print(f"Test  size : {len(te)}")
        print(f"Epochs     : {self.cfg.epoch}")
        print(f"Batch size : {self.cfg.bs}")
        print(f"Batches/ep : {total_batches}")
        print("-" * 50)
        sys.stdout.flush()

        e = 0
        while e < self.cfg.epoch:
            random.shuffle(tr)
            model.train()

            epoch_loss  = 0.0
            batch_count = 0
            i = 0

            bar = tqdm(
                total=total_batches,
                desc=f"Epoch {e+1:>2}/{self.cfg.epoch}",
                unit="batch",
                ncols=70,
                file=sys.stdout
            )

            while i < len(tr):
                idxs  = tr[i:i + self.cfg.bs]
                batch = data.get_batch(idxs)
                art   = en.tensor(batch.art_ids, dtype=en.long, device=dev)
                summ  = en.tensor(batch.sum_ids, dtype=en.long, device=dev)
                logp, cov = model.forward(art, summ)
                tgt   = summ[:, 1:]
                loss  = self.nll(logp, tgt)
                loss  = loss + self.cfg.cov * cov
                opt.zero_grad()
                loss.backward()

                import torch
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)

                opt.step()

                epoch_loss  += loss.item()
                batch_count += 1
                avg = epoch_loss / batch_count
                bar.set_postfix(loss=f"{avg:.4f}")
                bar.update(1)
                i += self.cfg.bs

            bar.close()
            avg_loss = epoch_loss / max(1, batch_count)
            print(f"  Epoch {e+1} done | Avg Loss: {avg_loss:.4f}")
            sys.stdout.flush()

            if e == self.cfg.epoch - 1:
                self.eval(model, data, tok, te, dev)

            e += 1

    def eval(self, model, data, tok, idxs, dev):
        model.eval()
        if len(idxs) == 0:
            print("\n  No test samples — skipping eval")
            return

        scorer = rs.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

        print("\n" + "=" * 50)
        print("  EVALUATION ON TEST SET")
        print("=" * 50)

        total_f1 = 0.0
        total_r1 = 0.0
        total_r2 = 0.0
        total_rl = 0.0
        j = 0
        k = 0

        while j < len(idxs) and k < 5:
            b     = [idxs[j]]
            batch = data.get_batch(b)
            art   = en.tensor(batch.art_ids, dtype=en.long, device=dev)
            pred  = model.generate(art, data.cfg.max_sum_len)
            pid   = pred[0].tolist()
            out   = tok.decode(pid)
            ref   = data.ex[idxs[j]].summ

            # existing word overlap metric
            pa, ra, fa = mt.overlap(out.split(), ref.lower().split())
            total_f1 += fa

            # ROUGE scores
            scores = scorer.score(ref, out)
            r1 = scores['rouge1'].fmeasure
            r2 = scores['rouge2'].fmeasure
            rl = scores['rougeL'].fmeasure
            total_r1 += r1
            total_r2 += r2
            total_rl += rl

            print(f"\n  Sample {k+1}")
            print(f"  Generated : {out}")
            print(f"  Reference : {ref[:100]}")
            print(f"  Precision : {pa:.4f} | Recall : {ra:.4f} | F1 : {fa:.4f}")
            print(f"  ROUGE-1   : {r1:.4f} | ROUGE-2 : {r2:.4f} | ROUGE-L : {rl:.4f}")

            j += 1
            k += 1

        if k > 0:
            print(f"\n  --- Averages over {k} samples ---")
            print(f"  Avg Word F1 : {total_f1/k:.4f}")
            print(f"  Avg ROUGE-1 : {total_r1/k:.4f}")
            print(f"  Avg ROUGE-2 : {total_r2/k:.4f}")
            print(f"  Avg ROUGE-L : {total_rl/k:.4f}")
        print("=" * 50)
        sys.stdout.flush()