import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_parameters(n_x, n_h, n_y, scale=1.0):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x) * scale
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * scale
    b2 = np.zeros((n_y, 1))
    return {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

def forward_propagation(X, parameters):
    W1, b1 = parameters["W1"], parameters["b1"]
    W2, b2 = parameters["W2"], parameters["b2"]
    Z1 = W1 @ X + b1
    A1 = sigmoid(Z1)
    Z2 = W2 @ A1 + b2
    A2 = sigmoid(Z2)
    return A2, {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

def compute_cost(A2, Y):
    m = Y.shape[1]
    eps = 1e-9
    return float(np.squeeze(-(1/m) * np.sum(Y*np.log(A2+eps) + (1-Y)*np.log(1-A2+eps))))

def backward_propagation(parameters, cache, X, Y):
    m = X.shape[1]
    W2 = parameters["W2"]
    A1, A2 = cache["A1"], cache["A2"]
    dZ2 = A2 - Y
    dW2 = (1/m) * (dZ2 @ A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = W2.T @ dZ2
    dZ1 = dA1 * A1 * (1 - A1)
    dW1 = (1/m) * (dZ1 @ X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)
    return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

def update_parameters(parameters, grads, learning_rate):
    parameters["W1"] -= learning_rate * grads["dW1"]
    parameters["b1"] -= learning_rate * grads["db1"]
    parameters["W2"] -= learning_rate * grads["dW2"]
    parameters["b2"] -= learning_rate * grads["db2"]
    return parameters

def train(X, Y, n_h, epochs=10000, learning_rate=1.0, print_cost=True):
    n_x, n_y = X.shape[0], Y.shape[0]
    parameters = initialize_parameters(n_x, n_h, n_y)
    cost_history = []
    for i in range(epochs):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backward_propagation(parameters, cache, X, Y)
        parameters = update_parameters(parameters, grads, learning_rate)
        cost_history.append(cost)
        if print_cost and i % 1000 == 0:
            print(f"Epoch {i}, cost: {cost:.6f}")
    return parameters, cost_history

def predict(X, parameters, threshold=0.5):
    A2, _ = forward_propagation(X, parameters)
    return (A2 > threshold).astype(int)

# ---- Train on XOR ----
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

parameters, cost_history = train(X, Y, n_h=4, epochs=10000, learning_rate=1.0)
predictions = predict(X, parameters)
print("\nPredictions:", predictions)
print("Ground truth:", Y)
print("Accuracy:", np.mean(predictions == Y) * 100, "%")