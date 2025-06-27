
# 🧠 Neuron Visual Trainer

A simple interactive Python application that visually demonstrates how a basic artificial neuron learns through training using gradient descent. Built using **Tkinter** for the GUI and **Matplotlib** for animated visualization.


## 📦 Features

- **Trainable single neuron** using mean squared error (MSE) and manual gradient descent.
- **Animated visualization** of the training process across epochs.
- **Interactive prediction** mode to test the trained neuron on custom input.
- Real-time display of:
  - Input values
  - Output
  - Weights
  - Bias
  - Loss per epoch

## 🖼️ Demo

<img src="https://github.com/yourusername/neuron-visual-trainer/raw/main/demo.gif" alt="Demo Animation" width="600"/>

> _Replace with your own GIF or screenshot if hosted._

---

## 🛠️ Installation

1. **Clone the repo:**
   ```bash
   git clone https://github.com/yourusername/neuron-visual-trainer.git
   cd neuron-visual-trtrainer
   ```

2. **Install dependencies:**
   ```bash
   pip install matplotlib
   ```

   Tkinter is usually included with Python. If not:
   - On Ubuntu: `sudo apt-get install python3-tk`
   - On Windows/macOS: Tkinter should come pre-installed with Python.

---

## 🚀 Usage

Run the application with:

```bash
python neuron_gui.py
```

### Inputs:
- Enter two input values in the fields.
- Click **Predict** to visualize the neuron's output.

---

## 🧠 How It Works

The neuron is defined as:

```
output = (input1 * w1) + (input2 * w2) + bias
```

The application uses a simple training dataset:

```python
training_data = [([1, 2], 3), ([0, 4], 4), ([3, 5], 8), ([2, 2], 4)]
```

Training uses:
- Mean Squared Error: `(output - target)^2`
- Manual gradient descent update for weights and bias.

The visualization shows how the weights, bias, and loss evolve during training.

---

## 📁 Project Structure

```bash
neuron-visual-trainer/
│
├── neuron_gui.py      # Main application script
├── README.md          # This file
```

---

## ✨ Future Ideas

- Add multi-layer (MLP) support.
- Live training instead of precomputed.
- Export final model weights.
- Enhanced visualization and color themes.

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).

---

## 🙋‍♂️ Contributors

[Vighneshwar Kuru](https://github.com/vighneshwarkuru)
[Sloka reddy](https://github.com/slokareddyyy)
