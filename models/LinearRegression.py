import numpy as np


class LinearRegression:

    def __init__(self):
        self.W = None
        self.B = None

    def train(self, x, y, epochs=1100):
        self.m, self.n = x.shape
        self.m = int(self.m)
        self.n = int(self.n)
        self.W = np.random.rand(self.n, y.shape[1])
        self.B = np.random.rand(self.m, y.shape[1])

        for i in range(epochs):
            y_pred = x * self.W + self.B
            L = np.sum(np.square(y_pred - y))
            dw = (1/self.m) * np.sum(np.multiply((y_pred - y), x))
            db = (1/self.m) * np.sum((y_pred - y))
            self.W -= 0.001 * dw
            self.B -= 0.001 * db
            if i % 10 == 0:
                print(f'Epoch {i}: loss = {L}')

    def predict(self, x):
        print(self.m, self.n)
        mat = np.zeros((self.m, self.n))
        mat = mat + x
        print(mat)
        y_pred = x * self.W + self.B
        print(y_pred, y_pred[0][0])



if __name__ == "__main__":
    x = np.array(range(10)).reshape(10, 1)
    y = np.array([0, 1]*5).reshape(10, 1)
    print(y)
    inst = LinearRegression()
    inst.train(x, y)
    inst.predict(np.array([1]))
