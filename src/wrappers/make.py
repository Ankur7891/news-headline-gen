from core.dataset import NewsData
from core.preprocess import WordTok
from core.rnn import RNNSeq
from core.lstm import LSTMSeq
from core.transformer import TransSeq
from core.trainer import Trainer


class DataBuild:
    def build(self, cfg):
        return NewsData(cfg)


class TokBuild:
    def build(self, cfg):
        return WordTok(cfg)


class ModelBuild:
    def build(self, cfg, vocab):
        if cfg.name == "rnn":
            return RNNSeq(cfg, vocab)
        if cfg.name == "lstm":
            return LSTMSeq(cfg, vocab)
        if cfg.name == "transformer":
            return TransSeq(cfg, vocab)
        raise ValueError("bad model")


class TrainBuild:
    def build(self, cfg):
        return Trainer(cfg)
