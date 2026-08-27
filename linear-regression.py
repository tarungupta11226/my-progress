import numpy as np

class LinearRegression:
    def __init__(self, lr=0.1, epochs=1000):
        self.lr = lr
        self.epochs = epochs
        self.w = None
        self.b = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.epochs):
            y_pred = X @ self.w + self.b
            error = y_pred - y
            loss = np.mean(error ** 2)

            dw = (2 / n_samples) * (X.T @ error)
            db = (2 / n_samples) * np.sum(error)

            self.w -= self.lr * dw
            self.b -= self.lr * db

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        def predict(self, X):
            return X @ self.w + self.b


np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 3 * X.flatten() + 5 + np.random.randn(100)  # add noise

model = LinearRegression(lr=0.01, epochs=1000)
model.fit(X, y)

print("Learned weight:", model.w)
print("Learned bias:", model.b)