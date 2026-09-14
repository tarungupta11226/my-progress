import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
X_raw, y_raw = data.data, data.target
feature_names = data.feature_names

counts = np.bincount(y_raw)
print(f"Class counts -> malignant(0): {counts[0]}, benign(1): {counts[1]}")
print("Any NaNs in X?", np.isnan(X_raw).any())
corr = np.corrcoef(X_raw[:, 0], y_raw)[0, 1]
print(f"Correlation of '{feature_names[0]}' with target: {corr:.3f}\n")

np.random.seed(42)
m_total = X_raw.shape[0]
perm = np.random.permutation(m_total)
train_end, dev_end = int(0.6 * m_total), int(0.8 * m_total)
train_idx, dev_idx, test_idx = perm[:train_end], perm[train_end:dev_end], perm[dev_end:]

X_train_raw, y_train = X_raw[train_idx], y_raw[train_idx]
X_dev_raw,   y_dev   = X_raw[dev_idx],   y_raw[dev_idx]
X_test_raw,  y_test  = X_raw[test_idx],  y_raw[test_idx]

mu = X_train_raw.mean(axis=0, keepdims=True)
sigma = X_train_raw.std(axis=0, keepdims=True) + 1e-9
standardize = lambda X: (X - mu) / sigma

X_train = standardize(X_train_raw).T
X_dev   = standardize(X_dev_raw).T
X_test  = standardize(X_test_raw).T
Y_train, Y_dev, Y_test = y_train.reshape(1, -1), y_dev.reshape(1, -1), y_test.reshape(1, -1)

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

def initialize_parameters_deep(layer_dims, seed=3):
    np.random.seed(seed)
    parameters = {}
    L = len(layer_dims) - 1
    for l in range(1, L + 1):
        parameters[f"W{l}"] = np.random.randn(layer_dims[l], layer_dims[l-1]) * np.sqrt(2.0 / layer_dims[l-1])
        parameters[f"b{l}"] = np.zeros((layer_dims[l], 1))
    return parameters

def linear_forward(A_prev, W, b):
    Z = W @ A_prev + b
    return Z, (A_prev, W, b)

def linear_activation_forward(A_prev, W, b, activation):
    Z, linear_cache = linear_forward(A_prev, W, b)
    A, activation_cache = (sigmoid(Z) if activation == "sigmoid" else relu(Z))
    return A, (linear_cache, activation_cache)

def L_model_forward(X, parameters):
    caches = []
    A = X
    L = len(parameters) // 2
    for l in range(1, L):                       
        A, cache = linear_activation_forward(A, parameters[f"W{l}"], parameters[f"b{l}"], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], "sigmoid")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, parameters=None, lambd=0.0):
    m = Y.shape[1]
    eps = 1e-9
    cost = -(1/m) * np.sum(Y*np.log(AL+eps) + (1-Y)*np.log(1-AL+eps))
    if parameters is not None and lambd > 0:     # L2 regularization term
        L = len(parameters) // 2
        l2 = sum(np.sum(np.square(parameters[f"W{l}"])) for l in range(1, L+1))
        cost += (lambd / (2*m)) * l2
    return float(np.squeeze(cost))

def linear_backward(dZ, cache, lambd=0.0):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * (dZ @ A_prev.T)
    if lambd > 0:
        dW += (lambd/m) * W                      
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = W.T @ dZ
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation, lambd=0.0):
    linear_cache, activation_cache = cache
    dZ = relu_backward(dA, activation_cache) if activation == "relu" else sigmoid_backward(dA, activation_cache)
    return linear_backward(dZ, linear_cache, lambd)

def L_model_backward(AL, Y, caches, lambd=0.0):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape)
    eps = 1e-9
    dAL = -(np.divide(Y, AL+eps) - np.divide(1-Y, 1-AL+eps))

    current_cache = caches[L-1]                  
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = dA_prev, dW, db

    for l in reversed(range(L-1)):               
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads[f"dA{l+1}"], current_cache, "relu", lambd)
        grads[f"dA{l}"], grads[f"dW{l+1}"], grads[f"db{l+1}"] = dA_prev, dW, db

    return grads

def update_parameters(parameters, grads, learning_rate):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters[f"W{l}"] -= learning_rate * grads[f"dW{l}"]
        parameters[f"b{l}"] -= learning_rate * grads[f"db{l}"]
    return parameters

def train(X, Y, layer_dims, epochs=3000, learning_rate=0.05, lambd=0.0,
          X_dev=None, Y_dev=None, print_cost=True, seed=3):
    parameters = initialize_parameters_deep(layer_dims, seed=seed)
    history = {"train": [], "dev": []}
    for i in range(epochs):
        AL, caches = L_model_forward(X, parameters)          # predict
        cost = compute_cost(AL, Y, parameters, lambd)          # loss
        grads = L_model_backward(AL, Y, caches, lambd)         # gradient
        parameters = update_parameters(parameters, grads, learning_rate)  # update
        history["train"].append(cost)
        if X_dev is not None:
            AL_dev, _ = L_model_forward(X_dev, parameters)
            history["dev"].append(compute_cost(AL_dev, Y_dev))
        if print_cost and i % 500 == 0:
            dev_str = f", dev_cost: {history['dev'][-1]:.4f}" if X_dev is not None else ""
            print(f"Epoch {i}, train_cost: {cost:.4f}{dev_str}")
    return parameters, history

def predict(X, parameters, threshold=0.5):
    AL, _ = L_model_forward(X, parameters)
    return (AL > threshold).astype(int), AL

n_x = X_train.shape[0]
layer_dims = [n_x, 16, 8, 1] 
print(f"Architecture: {layer_dims}\n")

parameters, history = train(X_train, Y_train, layer_dims, epochs=3000, learning_rate=0.05,
                             lambd=0.7, X_dev=X_dev, Y_dev=Y_dev, print_cost=True)

train_preds, _ = predict(X_train, parameters)
dev_preds, _ = predict(X_dev, parameters)
test_preds, test_probs = predict(X_test, parameters)

print(f"\nTrain accuracy: {np.mean(train_preds==Y_train)*100:.2f}%")
print(f"Dev accuracy:   {np.mean(dev_preds==Y_dev)*100:.2f}%")
print(f"Test accuracy:  {np.mean(test_preds==Y_test)*100:.2f}%")

def confusion_matrix_manual(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    TP = np.sum((y_true==1)&(y_pred==1)); TN = np.sum((y_true==0)&(y_pred==0))
    FP = np.sum((y_true==0)&(y_pred==1)); FN = np.sum((y_true==1)&(y_pred==0))
    return TP, TN, FP, FN

def print_metrics(y_true, y_pred, name):
    TP, TN, FP, FN = confusion_matrix_manual(y_true, y_pred)
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = TP/(TP+FP) if (TP+FP) > 0 else 0
    recall = TP/(TP+FN) if (TP+FN) > 0 else 0
    f1 = 2*precision*recall/(precision+recall) if (precision+recall) > 0 else 0
    print(f"\n{name}: TP={TP} TN={TN} FP={FP} FN={FN}")
    print(f"  Accuracy={accuracy*100:.2f}%  Precision={precision*100:.2f}%  Recall={recall*100:.2f}%  F1={f1*100:.2f}%")

print_metrics(Y_test, test_preds, "Neural Network — Test Set")
y_test_flat, pred_flat, prob_flat = Y_test.flatten(), test_preds.flatten(), test_probs.flatten()
misclassified = np.where(pred_flat != y_test_flat)[0]
print(f"\nMisclassified: {len(misclassified)}/{len(y_test_flat)}")
for idx in misclassified:
    print(f"  idx={idx}: true={y_test_flat[idx]}, pred={pred_flat[idx]}, prob={prob_flat[idx]:.3f}")