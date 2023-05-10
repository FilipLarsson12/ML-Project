import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


x = np.arange(-1, 1.01, 0.01).reshape(1,-1)


# Denna funktion skapar en plot som illustrerar vars sannolikhetsmassan ligger för värdena på w0 och w1:

def plot_w_prior():
    w_dimension = 2
    alpha = 0.2
    mean = np.zeros(w_dimension)
    covariance = np.eye(w_dimension) / alpha
    prior_w = multivariate_normal(mean, covariance)
    x, y = np.mgrid[-2:2:.01, -2:2:.01]
    pos = np.dstack((x, y))

    # Compute the density function for each point in the grid
    density = prior_w.pdf(pos)

    # Make a contour plot
    plt.contourf(x, y, density, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title('Contour plot of Multivariate Normal Distribution')
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.show()


def main():
    plot_w_prior()


main()