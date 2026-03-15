# Codebase Overview

This folder is the full project source for News Headline Generation using RNN, LSTM, and Transformer models. The design is intentionally simple and academic. All high level files only import our own modules. External libraries are hidden in wrappers.

## How It Runs

1. Data is read from dataset/news.csv
2. Text is cleaned and tokenized into word ids
3. A model is built using a config
4. Training runs with teacher forcing and coverage loss
5. Inference generates a headline from the article

The entry point is run.py in the project root.

## Folder Map

core
Contains the main classes. This is where the models, dataset, tokenizer, and trainer live.

- base.py: base classes for data, tokenizer, model, and trainer
- dataset.py: loads news.csv and builds examples and batches
- preprocess.py: word tokenizer with PAD, SOS, EOS, UNK
- rnn.py: RNN encoder decoder with attention and coverage
- lstm.py: LSTM encoder decoder with attention and coverage
- transformer.py: encoder decoder transformer for generation
- trainer.py: training loop and simple evaluation

dtypes
Small data and config classes.

- config.py: dataset, tokenizer, model, and training configs
- records.py: Example and Batch containers
- enums.py: ModelName enum for model selection

utils
Small helpers that do not depend on external libraries.

- text.py: text normalize, split, join
- metrics.py: overlap score for quick checks
- seed.py: seed control through the engine wrapper

wrappers
Thin wrappers around external libraries. The rest of the project does not import torch or pandas directly.

- engine.py: tensor ops and layers
- table.py: csv loading
- make.py: builder classes that wire components

## How To Switch Models

In run.py, change ModelConfig to use a different model:

- ModelName.RNN
- ModelName.LSTM
- ModelName.TRANSFORMER

## Usage

Run training from the project root:

python run.py

Use the notebook workflow:

1. Open main.ipynb
2. Run the cells in order
3. Change ModelName in the config cell to switch models

## Notes

The code is short by design and meant for understanding core ideas, not for production scale.
