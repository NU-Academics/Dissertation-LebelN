# Dissertation-LebelN

**Data-Driven Resilience Analytics for Distributed Data Mesh Systems**

Author: Nathan Lebel

## Overview

This repository contains the code and analysis for a dissertation investigating
failure prediction and resource optimization in distributed systems using the
Google Cluster Traces v3 (2019) and Backblaze Hard Drive datasets.

## Current Phase: A (Explore)

Phase A focuses on data understanding through exploratory data analysis (EDA).
The primary work products are Jupyter notebooks (stored as Jupytext `.py` files)
and BigQuery cache queries.

## Environment

- **Local Development:** IntelliJ IDEA Ultimate (linting, editing, git)
- **Execution:** Google Colab (T4 GPU, BigQuery access)
- **Data Storage:** BigQuery (cached tables) + Google Drive (checkpoints, outputs)
- **Language:** Python 3.12
- **DataFrame Library:** Polars (not Pandas)

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR-USERNAME/Dissertation-LebelN.git
cd Dissertation-LebelN

# Install dependencies
pip install -r requirements.txt

# Open 00_setup_environment.py in Colab to get started
```

## Repository Structure

```
Dissertation-LebelN/
├── sql/                  # BigQuery cache + exploration queries
├── notebooks/            # Jupytext .py files (primary work product)
├── utils/                # Minimal shared utilities
└── outputs/              # Generated tables and figures
```
