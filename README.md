<<<<<<< HEAD
# computational-biology--streamlit-app
Computational Biology Research Platform An integrated platform that brings together molecular simulations, genomic analysis, and cheminformatics to drive cutting-edge research in computational biology.
# Computational Biology Research Platform - Complete Edition

An integrated platform for advanced computational biology research combining molecular simulations, genomic analysis, and cheminformatics.

## Features

- Protein structure parsing and manipulation with Biopython
- Molecular structure handling and visualization with RDKit
- Molecular dynamics analysis with MDAnalysis
- Optional molecular dynamics simulations using OpenMM
- Data processing with Pandas and NumPy

## Installation

```bash
pip install biopython pandas numpy requests mdanalysis
# For RDKit and OpenMM, follow official installation guides:
# https://www.rdkit.org/docs/Install.html
# http://openmm.org/install.html
=======
# Computational Biology Research Platform

An integrated platform for advanced computational biology research combining molecular simulations, genomic analysis, and cheminformatics.

## 🔬 Features

- Protein structure parsing and manipulation (Biopython)
- Molecular structure visualization (RDKit)
- Molecular dynamics analysis (MDAnalysis)
- Optional simulations using OpenMM
- Data processing with Pandas and NumPy
- Logging and organized result folders

## 📦 Installation Guide

Some packages require special handling to install properly, especially RDKit, OpenMM, and MDAnalysis. We recommend using **Conda** for the smoothest setup:

```bash
conda create -n molenv python=3.9 -y
conda activate molenv

# Install core scientific packages
conda install -c conda-forge rdkit openmm mdanalysis pandas numpy plotly -y

# Install PyTorch (choose appropriate command from https://pytorch.org)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# Install other Python packages with pip
pip install streamlit py3dmol stmol requests web3 ipfshttpclient meeko vina-python biopython


## 📂 Project Structure

```
project/
├── a-demo.py             # Main Python script
├── README.md             # Project description
├── requirements.txt      # Required Python packages
├── .gitignore            # Files and folders to ignore in Git
├── cbp_results_*         # Auto-created output directories
```
>>>>>>> 33cd535 (Add initial project files)
