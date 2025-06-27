import tkinter as tk
from tkinter import messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import animation

# Neuron definition
class Neuron:
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights
        self.history = []

    def forward(self, inputs):
        self.inputs = inputs
        self.output = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.output

    def compute_loss(self, target):
        return (self.output - target) ** 2

    def backward(self, target, lr=0.05):
        dL_dout = 2 * (self.output - target)
        dL_dw = [dL_dout * i for i in self.inputs]
        dL_db = dL_dout
        self.weights = [w - lr * dw for w, dw in zip(self.weights, dL_dw)]
        self.bias -= lr * dL_db

    def log_state(self, epoch, loss):
        self.history.append({
            "epoch": epoch,
            "w1": self.weights[0],
            "w2": self.weights[1],
            "b": self.bias,
            "loss": loss,
            "input": self.inputs,
            "output": self.output
        })

# Main GUI app
class NeuronApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Neuron Visual Trainer")

        # Set up neuron and training data
        self.neuron = Neuron(bias=0.0, weights=[0.0, 0.0])
        self.training_data = [([1, 2], 3), ([0, 4], 4), ([3, 5], 8), ([2, 2], 4)]
        self.train_neuron()

        # Plot area
        self.fig, self.ax = plt.subplots(figsize=(7, 5))
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=0, columnspan=4)

        # Entry fields
        tk.Label(root, text="Input 1:").grid(row=1, column=0)
        self.input1 = tk.Entry(root)
        self.input1.grid(row=1, column=1)

        tk.Label(root, text="Input 2:").grid(row=1, column=2)
        self.input2 = tk.Entry(root)
        self.input2.grid(row=1, column=3)

        self.predict_btn = tk.Button(root, text="Predict", command=self.predict)
        self.predict_btn.grid(row=2, column=0, columnspan=4)

        # Start training animation
        self.ani = animation.FuncAnimation(self.fig, self.animate_training, frames=len(self.neuron.history), interval=300, repeat=False)
        self.canvas.draw()

    def train_neuron(self):
        for epoch in range(50):
            total_loss = 0
            for inputs, target in self.training_data:
                output = self.neuron.forward(inputs)
                loss = self.neuron.compute_loss(target)
                self.neuron.backward(target)
                total_loss += loss
            self.neuron.log_state(epoch, total_loss)

    def draw_neuron(self, input1, input2, output, w1, w2, bias, loss, title):
        self.ax.clear()
        self.ax.axis('off')
        self.ax.set_title(title, fontsize=12)

        pos = {
            "i1": (0, 1),
            "i2": (0, -1),
            "n": (3, 0),
            "o": (6, 0),
            "b": (-1, 2)
        }

        # Neuron circles
        for k, p in pos.items():
            if k != 'b':
                self.ax.add_patch(plt.Circle(p, 0.4, color='lightblue', ec='black'))
        
        self.ax.text(*pos['i1'], f"{input1}", ha='center', va='center')
        self.ax.text(*pos['i2'], f"{input2}", ha='center', va='center')
        self.ax.text(*pos['n'], f"Out\n{output:.2f}", ha='center', va='center')
        self.ax.text(*pos['o'], f"→", fontsize=15, ha='center', va='center')

        # Connections
        def connect(p1, p2, w, color='gray'):
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color)
            mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            self.ax.text(mid[0], mid[1], f"{w:.2f}", fontsize=8)

        connect(pos['i1'], pos['n'], w1)
        connect(pos['i2'], pos['n'], w2)
        connect(pos['n'], pos['o'], 1.0)
        self.ax.plot([pos['b'][0], pos['n'][0]], [pos['b'][1], pos['n'][1]], 'r--')
        self.ax.text((pos['b'][0]+pos['n'][0])/2, (pos['b'][1]+pos['n'][1])/2, f"b: {bias:.2f}", color='red')

        self.ax.set_xlim(-2, 7)
        self.ax.set_ylim(-3, 3)

    def animate_training(self, i):
        state = self.neuron.history[i]
        self.draw_neuron(
            input1=state['input'][0],
            input2=state['input'][1],
            output=state['output'],
            w1=state['w1'],
            w2=state['w2'],
            bias=state['b'],
            loss=state['loss'],
            title=f"Training Epoch {state['epoch']}, Loss: {state['loss']:.2f}"
        )
        self.canvas.draw()

    def predict(self):
        try:
            x1 = float(self.input1.get())
            x2 = float(self.input2.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numbers.")
            return

        out = self.neuron.forward([x1, x2])
        final = self.neuron.history[-1]
        self.draw_neuron(
            input1=x1,
            input2=x2,
            output=out,
            w1=final['w1'],
            w2=final['w2'],
            bias=final['b'],
            loss=0,
            title="Prediction"
        )
        self.canvas.draw()

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = NeuronApp(root)
    root.mainloop()
