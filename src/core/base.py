class BaseData:
    def split(self):
        raise NotImplementedError()


class BaseTok:
    def build_vocab(self, xs):
        raise NotImplementedError()

    def encode(self, s, max_len):
        raise NotImplementedError()

    def decode(self, ids):
        raise NotImplementedError()


class BaseModel:
    def forward(self, a, b):
        raise NotImplementedError()

    def generate(self, a, max_len):
        raise NotImplementedError()


class BaseTrain:
    def fit(self, model, data, tok):
        raise NotImplementedError()
