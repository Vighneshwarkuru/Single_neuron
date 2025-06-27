class Neuron:
    def __init__(self, bias, weights, inputs):
        self.bias = bias
        self.weights = weights
        self.inputs = inputs
        self.output = 0  # Initialize output

    def forward(self):
        # Calculate the weighted sum + bias
        weighted_sum = sum([w * i for w, i in zip(self.weights, self.inputs)]) + self.bias
        self.output = weighted_sum
        return self.output

# Create an instance of Neuron with inputs, weights, and bias
mySingleNeuron = Neuron(
    bias=3,
    weights=[0.3, 0.5, 0.1],
    inputs=[1.0, 2.0, 3.0]  # Add example input values
)

# Call the forward method to calculate the output
output = mySingleNeuron.forward()
print(f"Output: {output}")
