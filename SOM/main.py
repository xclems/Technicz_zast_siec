import numpy as np
import matplotlib.pyplot as plt

class SOM:
    def __init__(self, grid_size, input_dim, lr=0.5, sigma=None):
        self.grid_size = grid_size
        self.input_dim = input_dim
        self.lr0 = lr
        self.sigma0 = sigma if sigma else grid_size / 2

        # веса нейронов: (N, N, input_dim)
        self.weights = np.random.rand(grid_size, grid_size, input_dim)

        # координаты нейронов в решетке
        self.coords = np.array([[i, j] for i in range(grid_size) for j in range(grid_size)])

    def _get_bmu(self, x):
        # расстояние до всех нейронов
        diff = self.weights - x
        dist = np.linalg.norm(diff, axis=2)
        return np.unravel_index(np.argmin(dist), dist.shape)

    def _neighborhood(self, bmu, sigma):
        # гауссово соседство
        ax = np.arange(self.grid_size)
        xx, yy = np.meshgrid(ax, ax)
        dist_sq = (xx - bmu[0])**2 + (yy - bmu[1])**2
        return np.exp(-dist_sq / (2 * sigma**2))

    def train(self, data, epochs=1000):
        for t in range(epochs):
            # случайный вход
            x = data[np.random.randint(len(data))]

            # decay
            lr = self.lr0 * np.exp(-t / epochs)
            sigma = self.sigma0 * np.exp(-t / epochs)

            # победитель
            bmu = self._get_bmu(x)

            # соседство
            h = self._neighborhood(bmu, sigma)

            # обновление весов
            for i in range(self.grid_size):
                for j in range(self.grid_size):
                    self.weights[i, j] += lr * h[i, j] * (x - self.weights[i, j])

    def plot(self, data=None):
        plt.figure(figsize=(6, 6))

        # линии решетки
        for i in range(self.grid_size):
            plt.plot(self.weights[i, :, 0], self.weights[i, :, 1], 'k-')
            plt.plot(self.weights[:, i, 0], self.weights[:, i, 1], 'k-')

        # точки данных
        if data is not None:
            plt.scatter(data[:, 0], data[:, 1], c='red', s=10, alpha=0.5)

        # нейроны
        plt.scatter(self.weights[:, :, 0], self.weights[:, :, 1], c='blue')

        plt.title("SOM (square grid)")
        plt.grid()
        plt.show()


def main():
    # данные: круг (как в статье)
    n = 500
    theta = np.linspace(0, 2*np.pi, n)
    r = 0.4
    data = np.stack([
        r*np.cos(theta) + 0.5,
        r*np.sin(theta) + 0.5
    ], axis=1)

    # создаем SOM
    som = SOM(grid_size=15, input_dim=2)

    # обучение
    som.train(data, epochs=20000)

    # визуализация
    som.plot(data)


main()