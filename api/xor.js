export default function handler(req, res) {
  res.send(`

# XOR Gate using Backpropagation Neural Network
import numpy as np

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

Y = np.array([[0],
              [1],
              [1],
              [0]])

# Sigmoid activation and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Initialize weights and biases
np.random.seed(42)
input_neurons = 2
hidden_neurons = 2
output_neurons = 1
learning_rate = 0.1
epochs = 10000

# Random initialization
w_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
b_hidden = np.random.uniform(size=(1, hidden_neurons))
w_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))
b_output = np.random.uniform(size=(1, output_neurons))

# Training via backpropagation
for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(X, w_input_hidden) + b_hidden
    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, w_hidden_output) + b_output
    final_output = sigmoid(final_input)

    # Compute error
    error = Y - final_output

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(w_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights and biases
    w_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    b_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    w_input_hidden += X.T.dot(d_hidden) * learning_rate
    b_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

    # Print loss occasionally
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}, Error: {np.mean(np.abs(error)):.4f}")

# Testing 
print("\nFinal outputs after training:")
for i in range(len(X)):
    hidden_layer = sigmoid(np.dot(X[i], w_input_hidden) + b_hidden)
    output_layer = sigmoid(np.dot(hidden_layer, w_hidden_output) + b_output)
    print(f"Input: {X[i]} → Predicted Output: {np.round(output_layer[0], 3)}")
    
    
#using sklearn NN 

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# Same XOR data
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0])

# Define model
model = MLPClassifier(hidden_layer_sizes=(2,), activation='logistic',
                      solver='sgd', learning_rate_init=0.1, max_iter=10000, random_state=42)

# Train model
model.fit(X, y)

# Predictions
y_pred = model.predict(X)
print("Predictions:", y_pred)
print("Accuracy:", accuracy_score(y, y_pred))


`);
}
