# Stack Pointer Network for CoNLL2018 Shared Task
This is the code for our paper: [Joint Learning of POS and Dependencies for Multilingual Universal Dependency Parsing]()

The code is based on [NeuroNLP2](https://github.com/XuezheMax/NeuroNLP2), we are very appreciate the contribution of Xuezhe Ma.

## Requirements

Python 2.7, PyTorch >=0.3.0, Gensim >= 0.12.0


## Running the experiments

### Universal Dependency Parsing
To train a Joint-POS-Stack-Pointer parser, simply run

    ./example/run_stackPtrParser.sh
Remeber to setup the paths for data and embeddings.