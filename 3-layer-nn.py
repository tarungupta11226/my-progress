import numpy as np
np.random.seed(1)

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A, Z

def relu(Z):
    A = np.maximum(0, Z)
    return A, Z

def sigmoid_backward(dA, Z):
    A = 1 / (1 + np.exp(-Z))
    return dA * A * (1 - A)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims) - 1

    for l in range(1, L + 1):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))

        assert parameters[f"W{l}"].shape == (layer_dims[l], layer_dims[l-1])
        assert parameters[f"b{l}"].shape == (layer_dims[l], 1)

    return parameters

def linear_forward(A_prev, W, b):
    Z = W @ A_prev + b
    cache = (A_prev, W, b)
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    if activation == "sigmoid":
        A, activation_cache = sigmoid(Z)
    if activation == "relu":
        A, activation_cache = relu(Z)
    cache = (linear_cache, activation_cache)
    return A, cache

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2

    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters[f"W{l}"], parameters[f"b{l}"], "relu")
        caches.append(cache)
    A_prev = A
    AL, cache = linear_activation_forward(A_prev, parameters[f"W{L}"], parameters[f"b{L}"], "sigmoid")
    caches.append(cache)

    assert AL.shape == (1, X.shape[1])
    return AL, caches

def compute_cost(AL, Y):
    m = Y.shape[1]
    eps = 1e-9
    cost = -(1/m) * np.sum(Y*np.log(AL+eps) + (1-Y)*np.log(1-AL+eps))
    return float(np.squeeze(cost))

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * (dZ @ A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    return dA_prev, dW, db

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    eps = 1e-9

    dAL = -(np.divide(Y, AL+eps) - np.divide(1-Y, 1-AL+eps))
    
    current_cache = caches[L-1]
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, activation="sigmoid")
    grads[f"dA{L-1}"] = dA_prev
    grads[f"dW{L}"] = dW
    grads[f"db{L}"] = db

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads[f"dA{l+1}"], current_cache, "relu")
        grads[f"dA{l}"] = dA_prev
        grads[f"dW{l+1}"] = dW
        grads[f"db{l+1}"] = db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

def train(X, Y, layer_dims, epochs=3000, learning_rate=0.1, print_cost=True):
    parameters = initialize_parameters_deep(layer_dims)
    cost_history = []
    for i in range(epochs):
        AL, caches = L_model_forward(X, parameters)
        cost = compute_cost(AL, Y)
        grads = L_model_backward(AL, Y, caches)
        parameters = update_parameters(parameters, grads, learning_rate)
        cost_history.append(cost)
        if print_cost and i % 500 == 0:
            print(f"Epoch {i}, cost: {cost:.6f}")
    return parameters, cost_history

def predict(X, parameters, threshold=0.5):
    AL, _ = L_model_forward(X, parameters)
    return (AL > threshold).astype(int)

def dictionary_to_vector(params_dict, L, keys=("W","b")):
    vec, shapes = [], {}
    for l in range(1, L+1):
        for k in keys:
            arr = params_dict[f"{k}{l}"]
            shapes[f"{k}{l}"] = arr.shape
            vec.append(arr.reshape(-1, 1))
    return np.concatenate(vec, axis=0), shapes

def vector_to_dictionary(vec, shapes, L, keys=("W","b")):
    params_dict = {}
    idx = 0
    for l in range(1, L+1):
        for k in keys:
            shape = shapes[f"{k}{l}"]
            size = shape[0]*shape[1]
            params_dict[f"{k}{l}"] = vec[idx:idx+size].reshape(shape)
            idx += size
    return params_dict

def gradient_check_deep(parameters, grads, X, Y, epsilon=1e-7):
    L = len(parameters) // 2
    param_vec, shapes = dictionary_to_vector(parameters, L)
    grad_dict = {f"W{l}": grads[f"dW{l}"] for l in range(1,L+1)}
    grad_dict.update({f"b{l}": grads[f"db{l}"] for l in range(1,L+1)})
    grad_vec, _ = dictionary_to_vector(grad_dict, L)

    n = param_vec.shape[0]
    numerical_grad = np.zeros((n,1))

    for i in range(n):
        theta_plus = np.copy(param_vec); theta_plus[i,0] += epsilon
        AL_plus,_ = L_model_forward(X, vector_to_dictionary(theta_plus, shapes, L))
        cost_plus = compute_cost(AL_plus, Y)

        theta_minus = np.copy(param_vec); theta_minus[i,0] -= epsilon
        AL_minus,_ = L_model_forward(X, vector_to_dictionary(theta_minus, shapes, L))
        cost_minus = compute_cost(AL_minus, Y)

        numerical_grad[i,0] = (cost_plus - cost_minus) / (2*epsilon)

    diff = np.linalg.norm(grad_vec - numerical_grad) / (np.linalg.norm(grad_vec) + np.linalg.norm(numerical_grad) + 1e-12)
    print(f"Gradient check relative difference: {diff:.2e}  {'PASS' if diff < 1e-6 else 'FAIL'}")

np.random.seed(1)
base_X = np.array([[0,0,1,1],[0,1,0,1]], dtype=float)
base_Y = np.array([[0,1,1,0]], dtype=float)

X = np.tile(base_X, (1, 20)) + np.random.randn(2, 80) * 0.05
Y = np.tile(base_Y, (1, 20))

layer_dims = [2, 4, 4, 1]   
parameters, cost_history = train(X, Y, layer_dims, epochs=3000, learning_rate=0.5)

preds = predict(X, parameters)
print("Training accuracy:", np.mean(preds == Y) * 100, "%")
AL, caches = L_model_forward(X, parameters)
grads = L_model_backward(AL, Y, caches)
gradient_check_deep(parameters, grads, X, Y)