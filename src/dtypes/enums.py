from enum import Enum


class ModelName(str, Enum):
    RNN = "rnn"
    LSTM = "lstm"
    TRANSFORMER = "transformer"
