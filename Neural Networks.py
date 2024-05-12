import numpy as np


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.01
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def forward(self, inputs):
        Z1 = np.dot(inputs,self.W1) + self.b1
        A1 = self.sigmoid(Z1)
        Z2 = np.dot(A1, self.W2) + self.b2
        A2 = self.sigmoid(Z2)
        self.A2 = A2
        self.A1 = A1
        return A1, A2

    def backward(self, inputs, targets, learning_rate):
        m = inputs.shape[0]
        dZ2 = self.A2 - targets
        dW2 = np.dot(np.transpose(self.A1), dZ2) / m
        db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
        dZ1 = np.dot(dZ2, np.transpose(self.W2)) * (1 - np.power(self.A1, 2))
        dW1 = np.dot(np.transpose(inputs), dZ1) / m
        db1 = np.sum(dZ1, axis = 1, keepdims = True) / m


        W1 = self.W1 - learning_rate * dW1
        b1 = self.b1 - learning_rate * db1
        W2 = self.W2 - learning_rate * dW2
        b2 = self.b2 - learning_rate * db2


# Example usage
# Define input, hidden, and output sizes
input_size = 2
hidden_size = 3
output_size = 1

# Create a neural network
nn = NeuralNetwork(input_size, hidden_size, output_size)

# Example input and targets
inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
targets = np.array([[0], [1], [1], [0]])

# Training loop
epochs = 10000
learning_rate = 0.1
for epoch in range(epochs):
    # Forward pass
    A1, A2 = nn.forward(inputs)
    output = A2
    
    # Backward pass
    nn.backward(inputs, targets, learning_rate)
    
    # Print loss
    loss = np.mean(np.square(targets - output))
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}')

# Test the trained model
test_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
predicted_output = nn.forward(test_input)
print("Predicted output after training:")
print(predicted_output)
