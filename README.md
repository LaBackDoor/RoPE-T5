# RoPE-T5: A T5 Model with Rotary Position Embeddings

This repository provides a from-scratch implementation of a T5-style transformer model that replaces the standard relative position biases with **Rotary Position Embeddings (RoPE)**. The model is pre-trained on the C4 dataset using a span corruption objective.

This project serves as a foundational building block for more advanced models like **[RoPE-ByT5](https://github.com/LaBackDoor/RoPE-ByT5)** and is inspired by the work done in [melmoth/ru-rope-t5-small-instruct](https://huggingface.co/melmoth/ru-rope-t5-small-instruct).

## ✨ Features

* **T5 with Rotary Position Embeddings (RoPE)**: A from-scratch implementation of the T5 architecture that natively uses RoPE for positional information, removing the need for relative attention biases.
* **Efficient Pre-training**: Utilizes the Hugging Face `Trainer` with an efficient, streaming-based data pipeline to process the large C4 dataset on the fly.
* **Span Corruption**: Employs the canonical text-to-text denoising objective from the original T5 paper for robust pre-training.
* **Performance Optimized**: Integrated with Flash Attention 2 for optimized training speed and reduced memory footprint.
* **Comprehensive Evaluation**: Includes scripts to evaluate the pre-trained model on zero-shot and fine-tuning benchmarks like GLUE, SQuAD, and CNN/DailyMail.
* **Modern Tooling**: Set up with `uv` for fast and reliable Python package management.

## 🚀 Setup and Installation

This project uses `uv` for package management. Follow these steps to set up the environment.

1.  **Create a Virtual Environment:**
    First, create and activate a new virtual environment using `uv`.
    ```bash
    uv venv
    source .venv/bin/activate
    ```

2.  **Install Core Build Dependencies:**
    Install `torch` and `setuptools` first. This is a necessary step to prepare for building packages like `flash-attn` from source.
    ```bash
    uv add torch setuptools
    ```

3.  **Sync Project Dependencies:**
    Use `uv sync` with the `--no-build-isolation` flag. This flag is crucial as it allows `flash-attn` to find the already-installed `torch` and build correctly.
    ```bash
    uv sync --no-build-isolation
    ```

## ▶️ How to Run

Make sure you run all commands from the root directory of the project.

1.  **Pre-training the Model**

    To start pre-training the `RoPE-T5` model from scratch on the C4 dataset, run the following command:

    ```bash
    python scripts/run_pretraining.py
    ```

    The script will handle initializing the model, streaming the dataset, and saving checkpoints to the c4-rope-t5-from-scratch-stream directory in your project root. Training progress is logged to Weights & Biases.


2.  **Evaluating the Model**

    After pre-training is complete, you can evaluate your model on various downstream tasks using the evaluation script:

    ```bash
    python scripts/run_evaluation.py
    ```

This script will:
* Load your pre-trained model from the output directory.
* Run a series of zero-shot evaluations on benchmarks like GLUE and SQuAD.
* Save the results and intermediate files to the `evaluation/` directory.