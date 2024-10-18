# ML Router

The **ML Router** is a machine learning-based tool designed for routing tasks in VLSI design. This tool enables training, evaluation, and generation of tilesets for various design experiments. Below are instructions on how to use the tool for training and evaluating your ML model, as well as how to generate tilesets and run benchmarks.

## Table of Contents
1. [Installation](#installation)
2. [Training](#training)
3. [Evaluation](#evaluation)
   - [Evaluate on Generated Tileset](#evaluate-on-generated-tileset)
   - [Evaluate on Generated Design](#evaluate-on-generated-design)
   - [Evaluate on Benchmark](#evaluate-on-benchmark)
4. [Generating Tilesets](#generating-tilesets)
5. [Running Experiments](#running-experiments)
6. [License](#license)

---

## Installation

1. Ensure you have Python installed along with the necessary dependencies (e.g., TensorFlow, PyTorch, scikit-learn, or any other required libraries for ML training and inference).
2. Clone this repository:

   ```bash
   git clone https://github.com/your-repo/ml-router.git
   cd ml-router
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Training

To train the ML Router model, run the `train_network.py` script. The model will be trained on the **tileset** (provided as a pickle file), which is hardcoded in the `train_router` function.

### Steps:
1. Navigate to the repository folder.
2. Run the following command to start training:

   ```bash
   python train_network.py
   ```

The training process will begin using the tileset defined within the `train_router` function.

---

## Evaluation

### Evaluate on Generated Tileset

If you want to evaluate the model on a newly generated tiled design:
1. Run a new training session (as described in the [Training](#training) section).
2. After the training is complete, check the `./images` folder for the generated tileset evaluation results.

### Evaluate on Generated Design

To evaluate the model on a generated design:
1. Open the `evaluate.py` script.
2. Uncomment the relevant lines for generated design evaluation.
3. Run the following command:

   ```bash
   python evaluate.py
   ```

### Evaluate on Benchmark

To evaluate the model on benchmark datasets:
1. Simply run the `evaluate.py` script as follows:

   ```bash
   python evaluate.py
   ```

---

## Generating Tilesets

To generate new tilesets for training or evaluation, use the `generate_trainset.py` script:

1. Run the following command to generate tilesets:

   ```bash
   python generate_trainset.py
   ```

The script will generate a new tileset and save it for use in training or evaluation.

---

## Running Experiments

For running different experiments or custom tasks, you can call the appropriate function from the `train_network.py` script. This allows flexibility in exploring different ML models or variations in training and evaluation.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
