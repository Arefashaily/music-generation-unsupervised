<div align="center">

# Unsupervised Symbolic Music Generation

**A comparative PyTorch study of an LSTM Autoencoder, a β-VAE and a conditioned Transformer for generating MIDI music**

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![miditok](https://img.shields.io/badge/Tokenization-REMI%20(miditok)-1F3A5F?style=flat)
![Status](https://img.shields.io/badge/Status-Proof%20of%20concept-orange?style=flat)

[Read the report (PDF)](report/Unsupervised_Symbolic_Music_Generation.pdf)

</div>

An end-to-end research pipeline that takes raw MIDI files through preprocessing, model training and symbolic music generation, with four approaches side by side. Built as a course project for **CSE425 at BRAC University** and documented in an IEEE-format report.

> **Project status:** this is a proof of concept that validates the pipeline on a small synthetic dataset (300 segments), not a trained music generator. See [Current limitations](#current-limitations) before judging the output.

## Pipeline

```mermaid
flowchart LR
    A[Raw MIDI] --> B[Parse notes]
    B --> C[Piano roll<br/>64 x 88]
    B --> D[REMI tokens]
    C --> E[LSTM Autoencoder]
    C --> F[Beta-VAE]
    D --> G[Conditioned Transformer]
    B --> H[Markov baseline]
    E --> I[Generated MIDI]
    F --> I
    G --> I
    H --> I
    I --> J[Evaluation utilities]
```

## Models

| Model | Input | Architecture | Loss | Role in the study |
|---|---|---|---|---|
| **LSTM Autoencoder** | Piano roll | 2-layer LSTM, 256 hidden units, 64-dim latent | Focal | Reconstruction baseline |
| **β-VAE** | Piano roll | 2-layer LSTM, 256 hidden units, 128-dim latent | Focal + β-weighted KL | Structured latent space for sampling and interpolation |
| **Transformer decoder** | REMI tokens | 6 layers, 8 attention heads, 256-dim, 6.6M parameters | Cross-entropy | Long-range sequence modeling with a conditioning label |
| **Markov chain** | Pitch sequence | First-order pitch transition table | None | Statistical baseline |

**Representations**
- **Piano roll:** binarized 64-step by 88-pitch windows at 16 frames per second, filtered for sparsity. Focal loss handles the heavy class imbalance of mostly silent frames.
- **Event tokens:** REMI tokenization through miditok, with a 512-token vocabulary and sequences of up to 512 tokens.
- **Conditioning:** ten labels taken from the MAESTRO year folders (2004 to 2018). These are recording years, used here as a stand-in for style.

An experimental reinforcement-learning fine-tuning module (a reward model with REINFORCE) lives in `src/models/diffusion.py`. It needs human listening-survey data and has not been run or evaluated, so it is not part of the results.

## Results from the report

Training ran on CPU to validate correctness on the synthetic dataset of 20 MIDI files (240 training and 60 validation segments).

| Training configuration (Table I) | Quick-run losses (Table II) |
|:---:|:---:|
| <img src="assets/report/table-1-hyperparameters.png" alt="Table I: training hyperparameters from the report" width="400"> | <img src="assets/report/table-2-losses.png" alt="Table II: training and validation losses from the report" width="400"> |

The tables show the report's configured epochs. The saved quick-run checkpoints reached their best validation loss early:

| Model | Best epoch | Best validation loss |
|---|:---:|:---:|
| LSTM Autoencoder | 4 | 0.0752 |
| β-VAE | 5 | 0.0772 |
| Transformer | 2 | 4.1395 (perplexity 62.8) |

## Current limitations

Results here are a pipeline check, not evidence of musical quality:

- **Small synthetic data:** the models saw 300 segments of generated chord patterns, far smaller than the roughly 15,000 segments expected from MAESTRO.
- **Under-trained deep models:** sampling from the saved checkpoints gives degenerate output. The Autoencoder and VAE produce near-constant clusters of keys, and the Transformer produces very few notes. The Markov baseline produces usable note sequences.
- **Next step:** train on the full [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) dataset on a GPU, then evaluate with the included pitch-histogram and rhythm utilities and a human listening study.

## What this project demonstrates

- Designing a modular ML codebase with separate preprocessing, model, training, generation and evaluation packages
- Implementing LSTM autoencoders, variational autoencoders with KL annealing, and a conditioned Transformer in PyTorch
- Handling symbolic music as both image-like piano rolls and sequential event tokens
- Reporting a study's scope and limits clearly in an IEEE-format paper

## Getting started

```bash
git clone https://github.com/Arefashaily/music-generation-unsupervised.git
cd music-generation-unsupervised
pip install -r requirements.txt
```

Train each model on the included synthetic data:

```bash
python -m src.training.train_ae
python -m src.training.train_vae
python -m src.training.train_transformer
```

Generate MIDI files:

```bash
python -m src.generation.generate_music --model ae --n 5
python -m src.generation.generate_music --model vae --n 5
python -m src.generation.generate_music --model transformer --n 5 --genre 0
python -m src.generation.generate_music --model markov --n 5
```

Model checkpoints are not stored in this repository because of their size, so run training before generating with the neural models. Settings live in `src/config.py`. To use real data, place MAESTRO MIDI files in `data/raw_midi/<year>/`.

`pretty_midi.py` in the project root is a small local compatibility layer built on mido, added because the `pretty_midi` package failed to import on Python 3.13.

## Project structure

```
music-generation-unsupervised/
├── src/
│   ├── config.py          # Paths and hyperparameters
│   ├── preprocessing/     # MIDI parsing, piano rolls, REMI tokenization
│   ├── models/            # Autoencoder, VAE, Transformer, experimental RL
│   ├── training/          # Training loops
│   ├── generation/        # Sampling and MIDI export
│   └── evaluation/        # Pitch-histogram, rhythm and dataset metrics
├── data/                  # Synthetic MIDI files and train/validation splits
├── notebooks/             # Preprocessing and Markov baseline notebooks
├── report/                # IEEE-format report: PDF and LaTeX source
├── assets/report/         # Images taken from the report
├── pretty_midi.py
└── requirements.txt
```

## Report

| Page 1 | Page 2 | Page 3 | Page 4 |
|:---:|:---:|:---:|:---:|
| [![Report page 1](assets/report/page-1.png)](report/Unsupervised_Symbolic_Music_Generation.pdf) | [![Report page 2](assets/report/page-2.png)](report/Unsupervised_Symbolic_Music_Generation.pdf) | [![Report page 3](assets/report/page-3.png)](report/Unsupervised_Symbolic_Music_Generation.pdf) | [![Report page 4](assets/report/page-4.png)](report/Unsupervised_Symbolic_Music_Generation.pdf) |

## Acknowledgments

The target dataset is [MAESTRO v3.0.0](https://magenta.tensorflow.org/datasets/maestro) (Hawthorne et al., ICLR 2019), which is not redistributed here. Tokenization uses [miditok](https://github.com/Natooz/MidiTok).
