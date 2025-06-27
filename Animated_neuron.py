import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Neuron:
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights
        self.history = []  # Save training states

    def forward(self, inputs):
        self.inputs = inputs
        self.output = sum(w * i for w, i in zip(self.weights, inputs)) + self.bias
        return self.output

    def compute_loss(self, target):
        return (self.output - target) ** 2

    def backward(self, target, lr=0.01):
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

# --- Training ---
neuron = Neuron(bias=0.0, weights=[0.0, 0.0])

training_data = [
    ([1, 2], 3),
    ([0, 4], 4),
    ([3, 5], 8),
    ([2, 2], 4),
]

for epoch in range(50):  # Fewer epochs for a shorter animation
    total_loss = 0
    for inputs, target in training_data:
        output = neuron.forward(inputs)
        loss = neuron.compute_loss(target)
        neuron.backward(target, lr=0.05)
        total_loss += loss
    neuron.log_state(epoch, total_loss)

# --- Animation ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.axis('off')

positions = {
    "input1": (0, 1),
    "input2": (0, -1),
    "neuron": (4, 0),
    "output": (7, 0),
    "bias": (-1, 2)
}

def draw_frame(state):
    ax.clear()
    ax.axis('off')
    ax.set_title(f"Epoch {state['epoch']}, Loss: {state['loss']:.2f}", fontsize=14)

    # Neurons
    for name, pos in positions.items():
        if name != "bias":
            circle = plt.Circle(pos, 0.4, color='lightblue', ec='black')
            ax.add_patch(circle)

    # Labels
    ax.text(*positions["input1"], f"Input 1\n{state['input'][0]:.1f}", ha='center', va='center', fontsize=9)
    ax.text(*positions["input2"], f"Input 2\n{state['input'][1]:.1f}", ha='center', va='center', fontsize=9)
    ax.text(*positions["neuron"], f"Neuron\nOut: {state['output']:.2f}", ha='center', va='center', fontsize=9)
    ax.text(*positions["output"], "Output", ha='center', va='center', fontsize=9)

    # Connections
    def connect(p1, p2, weight, label):
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'gray')
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        ax.text(mid[0], mid[1], f"{weight:.2f}", fontsize=8, color='green')

    connect(positions["input1"], positions["neuron"], state["w1"], "w1")
    connect(positions["input2"], positions["neuron"], state["w2"], "w2")
    connect(positions["neuron"], positions["output"], 1.0, "")  # Output arrow

    # Bias line
    ax.plot([positions["bias"][0], positions["neuron"][0]], [positions["bias"][1], positions["neuron"][1]], 'r--')
    midb = ((positions["bias"][0] + positions["neuron"][0]) / 2, (positions["bias"][1] + positions["neuron"][1]) / 2)
    ax.text(*midb, f"b: {state['b']:.2f}", color='red', fontsize=8)
    ax.text(*positions["bias"], "Bias", fontsize=8)

    ax.set_xlim(-2, 8)
    ax.set_ylim(-3, 3)

# Animate
def animate(i):
    draw_frame(neuron.history[i])

ani = animation.FuncAnimation(fig, animate, frames=len(neuron.history), interval=400, repeat=False)

# --- Display or Save ---
plt.tight_layout()
plt.show()

# To save as MP4:
# ani.save("neuron_training.mp4", writer='ffmpeg', fps=2)

# To save as GIF (if you have imagemagick installed):
# ani.save("neuron_training.gif", writer='imagemagick', fps=2)
