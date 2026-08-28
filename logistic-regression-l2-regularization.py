import numpy as np

class LogisticRegressionL2:
    def __init__(self, lr=0.01, epochs=1000, lambda_=0.1):
        self.lr = lr
        self.epochs = epochs
        self.lambda_ = lambda_
        self.w = None
        self.b = None
        self.cost_history = []

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            z = X @ self.w + self.b
            y_pred = self.sigmoid(z)

            error = y_pred - y

            eps = 1e-9
            cross_entropy = -np.mean(y * np.log(y_pred + eps) + (1 - y) * np.log(1 - y_pred + eps))
            l2_penalty = (self.lambda_ / (2 * m)) * np.sum(self.w ** 2)
            cost = cross_entropy + l2_penalty
            self.cost_history.append(cost)

            dw = (1 / m) * (X.T @ error) + (self.lambda_ / m) * self.w
            db = (1 / m) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Cost: {cost:.4f}")

    def predict_proba(self, X):
        return self.sigmoid(X @ self.w + self.b)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
