import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


class AnharmonicOscillator:

    def __init__(self, interval_length=0.001, data_size=2000, boundary=0.5, omega=1, h_bar=1, m=1, coefficient=None):
        if coefficient is None:
            coefficient = [0, 0, 0]
        self.interval_length = interval_length
        self.data_size = data_size
        self.boundary = boundary
        self.omega = omega
        self.h_bar = h_bar
        self.m = m
        self.coefficient = coefficient

        self.position = np.linspace(-self.boundary, self.boundary, self.data_size)
        self.potential = np.zeros((self.data_size, self.data_size))
        self.eigenvalues = []
        self.eigenvectors = []
        self.__solve()

    def __solve(self):
        kinetic = 2 * np.eye(self.data_size) - 1 * np.eye(self.data_size, k=1) - 1 * np.eye(self.data_size, k=-1)
        kinetic_multiplier = self.h_bar ** 2 / (2 * self.m * self.interval_length ** 2)
        kinetic = kinetic_multiplier * kinetic

        for i in range(len(self.potential)):
            self.potential[i][i] = 0.5 * self.m * self.omega ** 2 * self.position[i] ** 2
            self.potential[i][i] += self.coefficient[0] * self.position[i] ** 3
            self.potential[i][i] += self.coefficient[1] * self.position[i] ** 4
            self.potential[i][i] += self.coefficient[2] * self.position[i] ** 5
        hamiltonian = kinetic + self.potential

        eigenvalues, eigenvectors = np.linalg.eig(hamiltonian)
        df = pd.DataFrame(data=np.transpose(eigenvectors), index=eigenvalues)
        df.sort_index(inplace=True)
        self.eigenvalues = df.index
        self.eigenvectors = df.values

    def show_figure_level(self, level=0):
        probability_density = self.eigenvectors[level] ** 2
        probability_density_norm = probability_density / self.interval_length
        plt.plot(self.position, probability_density_norm)
        plt.show()

    def get_pdf(self, level=0):
        probability_density = self.eigenvectors[level] ** 2
        return probability_density

    def get_pdf_norm_graph(self, level=0):
        probability_density = self.eigenvectors[level] ** 2
        probability_density_norm = probability_density / sum(probability_density * self.interval_length)
        return probability_density_norm
