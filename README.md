# Braindecoder

Video demo:

[![Video Thumbnail](https://img.youtube.com/vi/UqvO8JjcLMM/maxresdefault.jpg)](https://www.youtube.com/watch?v=UqvO8JjcLMM)


An experimental project exploring EEG-based latent space representations during sleep using variational autoencoders. 

# About

This project attempts to analyze EEG patterns during sleep using deep learning approaches, specifically:

VAE architecture for latent space encoding
Pink noise loss for brain-like representations
Visualization of latent space patterns

# Requirements

Python 3.8+

See requirements.txt for dependencies

# Setup
 
Clone the repository

Install dependencies: pip install -r requirements.txt

Run the app: 

python app.py

# Current Status

This is experimental software that:

Can process EEG sleep data

Generates latent space representations

Visualizes temporal patterns

Does not yet reliably generate clear images

# Usage

Basic usage example:

pythonCopypython Dream2Imagev13.py --config config.yaml

# Acknowledgments

Inspired by recent work in brain decoding, particularly Meta's research on MEG-based image reconstruction.
