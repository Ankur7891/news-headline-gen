from dtypes.config import DatasetConfig, TokenConfig, ModelConfig, TrainConfig
from dtypes.enums import ModelName
from wrappers.make import DataBuild, TokBuild, ModelBuild, TrainBuild


def main():
    dcfg = DatasetConfig("dataset/news.csv")
    tcfg = TokenConfig(1)
    mcfg = ModelConfig(ModelName.LSTM, 128, 256, 1, 4, 256)
    trcfg = TrainConfig(8, 0.001, 1, 1.0, 0.5, 7)
    data = DataBuild().build(dcfg)
    tok = TokBuild().build(tcfg)
    tok.build_vocab(data.vocab_texts())
    model = ModelBuild().build(mcfg, tok.size())
    tr = TrainBuild().build(trcfg)
    tr.fit(model, data, tok)


if __name__ == "__main__":
    main()
