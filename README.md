# Neural Networks From Scratch

This repository contains simple implementations of feedforward neural networks built entirely from scratch in Python — **no libraries** like TensorFlow or PyTorch used.

## 🧠 Models

### `threeNN.py` (3 Neuron Network)
- Architecture: 2 hidden neurons → 1 output neuron
- Task: Predicts whether a point is **above or below** the line `y = 2x + 0.3`
- Uses Mean Squared Error (MSE) and gradient descent

### `fiveNN.py` (5 Neuron Network)
- Architecture: 4 hidden neurons → 1 output neuron
- Deeper model for better approximation
- Manual backpropagation implemented step-by-step

## ⚙️ How to Run

```bash
python NN_2_1.py
python NN_2_2.py
