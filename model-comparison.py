import numpy as np
import time
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

data = load_breast_cancer()
X_raw, y_raw = data.data, data.target

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

X_train_std = standardize(X_train_raw)
X_dev_std   = standardize(X_dev_raw)
X_test_std  = standardize(X_test_raw)

results = {}

def evaluate(name, model, X_tr, y_tr, X_te, y_te, proba_fn=None):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)

    proba_te = proba_fn(X_te) if proba_fn is not None else model.predict_proba(X_te)[:, 1]

    results[name] = {
        "train_acc": accuracy_score(y_tr, pred_tr),
        "test_acc": accuracy_score(y_te, pred_te),
        "precision": precision_score(y_te, pred_te),
        "recall": recall_score(y_te, pred_te),
        "f1": f1_score(y_te, pred_te),
        "roc_auc": roc_auc_score(y_te, proba_te),
        "train_time": train_time,
        "gap": accuracy_score(y_tr, pred_tr) - accuracy_score(y_te, pred_te),
    }

evaluate("Logistic Regression", LogisticRegression(max_iter=2000, random_state=42),
          X_train_std, y_train, X_test_std, y_test)

evaluate("Decision Tree", DecisionTreeClassifier(max_depth=4, random_state=42),
          X_train_std, y_train, X_test_std, y_test)

evaluate("Random Forest", RandomForestClassifier(n_estimators=200, max_depth=4, random_state=42),
          X_train_std, y_train, X_test_std, y_test)

evaluate("Gradient Boosting", GradientBoostingClassifier(n_estimators=100, max_depth=3,
          learning_rate=0.1, random_state=42),
          X_train_std, y_train, X_test_std, y_test)

def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z)); return A, Z

def relu(Z):
    A = np.maximum(0, Z); return A, Z

def sigmoid_backward(dA, Z):
    A = 1 / (1 + np.exp(-Z)); return dA * A * (1 - A)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True); dZ[Z <= 0] = 0; return dZ

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
    for l in range(1, L):              # hidden layers: ReLU
        A, cache = linear_activation_forward(A, parameters[f"W{l}"], parameters[f"b{l}"], "relu")
        caches.append(cache)
    AL, cache = linear_activation_forward(A, parameters[f"W{L}"], parameters[f"b{L}"], "sigmoid")  # output: sigmoid
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y, parameters=None, lambd=0.0):
    m = Y.shape[1]; eps = 1e-9
    cost = -(1/m) * np.sum(Y*np.log(AL+eps) + (1-Y)*np.log(1-AL+eps))
    if parameters is not None and lambd > 0:          # L2 regularization term
        L = len(parameters) // 2
        l2 = sum(np.sum(np.square(parameters[f"W{l}"])) for l in range(1, L+1))
        cost += (lambd / (2*m)) * l2
    return float(np.squeeze(cost))

def linear_backward(dZ, cache, lambd=0.0):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m) * (dZ @ A_prev.T)
    if lambd > 0:
        dW += (lambd/m) * W                            # weight decay
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
    Y = Y.reshape(AL.shape); eps = 1e-9
    dAL = -(np.divide(Y, AL+eps) - np.divide(1-Y, 1-AL+eps))

    current_cache = caches[L-1]                        # output layer (sigmoid)
    dA_prev, dW, db = linear_activation_backward(dAL, current_cache, "sigmoid", lambd)
    grads[f"dA{L-1}"], grads[f"dW{L}"], grads[f"db{L}"] = dA_prev, dW, db

    for l in reversed(range(L-1)):                      # hidden layers (relu), backward
        current_cache = caches[l]
        dA_prev, dW, db = linear_activation_backward(grads[f"dA{l+1}"], current_cache, "relu", lambd)
        grads[f"dA{l}"], grads[f"dW{l+1}"], grads[f"db{l+1}"] = dA_prev, dW, db

    return grads

def update_parameters(parameters, grads, lr):
    L = len(parameters) // 2
    for l in range(1, L+1):
        parameters[f"W{l}"] -= lr * grads[f"dW{l}"]
        parameters[f"b{l}"] -= lr * grads[f"db{l}"]
    return parameters

def train_nn(X, Y, layer_dims, epochs=3000, lr=0.05, lambd=0.7, seed=3):
    parameters = initialize_parameters_deep(layer_dims, seed=seed)
    for i in range(epochs):
        AL, caches = L_model_forward(X, parameters)     # predict
        grads = L_model_backward(AL, Y, caches, lambd)  # gradient
        parameters = update_parameters(parameters, grads, lr)  # update
    return parameters

def nn_predict(X, parameters, threshold=0.5):
    AL, _ = L_model_forward(X, parameters)
    return (AL > threshold).astype(int).flatten(), AL.flatten()

X_train_nn, X_test_nn = X_train_std.T, X_test_std.T
Y_train_nn = y_train.reshape(1, -1)
n_x = X_train_nn.shape[0]

t0 = time.time()
nn_params = train_nn(X_train_nn, Y_train_nn, [n_x, 16, 8, 1], epochs=3000, lr=0.05, lambd=0.7)
nn_time = time.time() - t0

nn_pred_train, _ = nn_predict(X_train_nn, nn_params)
nn_pred_test, nn_proba_test = nn_predict(X_test_nn, nn_params)

results["Neural Network"] = {
    "train_acc": accuracy_score(y_train, nn_pred_train),
    "test_acc": accuracy_score(y_test, nn_pred_test),
    "precision": precision_score(y_test, nn_pred_test),
    "recall": recall_score(y_test, nn_pred_test),
    "f1": f1_score(y_test, nn_pred_test),
    "roc_auc": roc_auc_score(y_test, nn_proba_test),
    "train_time": nn_time,
    "gap": accuracy_score(y_train, nn_pred_train) - accuracy_score(y_test, nn_pred_test),
}

print(f"{'Model':<20}{'Train':<8}{'Test':<8}{'Prec':<8}{'Recall':<8}{'F1':<8}{'ROC-AUC':<9}{'Gap':<8}{'Time(s)':<8}")
for name, r in results.items():
    print(f"{name:<20}{r['train_acc']*100:<8.2f}{r['test_acc']*100:<8.2f}{r['precision']*100:<8.2f}"
          f"{r['recall']*100:<8.2f}{r['f1']*100:<8.2f}{r['roc_auc']*100:<9.2f}{r['gap']*100:<8.2f}{r['train_time']:<8.3f}")