# Fruits classification

Convolutional neural network for classifying fruit and vegetable images into ten categories (subset of [Fruits-360](https://www.kaggle.com/datasets/moltean/fruits)), using TensorFlow/Keras.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt
```

Download the Fruits-360 dataset and point `--train-dir` and `--test-dir` at its `Training` and `Test` folders (with subfolders named like `Apple`, `Banana`, …).

## Run training

```bash
python main.py --train-dir path/to/Training --test-dir path/to/Test --epochs 10
```

Optional arguments:

- `--checkpoint` — prefix path for weight checkpoints (default: `fruits_model_weights`)
- `--save-model` — SavedModel output directory (default: `fruits_saved_model`)
- `--batch-size` — batch size (default: 32)

After training, learning curves are written to `accuracy_curve.png` and `loss_curve.png`.

## Notebook

`Classification.ipynb` is the original Colab workflow (Google Drive paths). For local use, change `train_dir` and `test_dir` to your dataset paths. The VGG-style block in the notebook is **commented out** so it is not stacked on top of the first CNN by mistake.

## License

Include Fruits-360 attribution if you redistribute the dataset; see the dataset page on Kaggle.
