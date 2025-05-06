import numpy as np
import scipy.stats as stats
import math
import matplotlib.pyplot as plt

def fit_7D_gaussian(data, num_bins=100):
    """
    Fit a 7D Gaussian distribution to the given data.
    
    Parameters:
        data (numpy.ndarray): The input data of shape (N, 7).
        num_bins (int): The number of bins for histogram.
        
    Returns:
        tuple: A tuple containing the mean and covariance matrix of the fitted Gaussian.
    """
    # Calculate mean and covariance
    mean = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    
    return mean, cov

def plot_7D_gaussian(mean, cov, num_samples=1000, data=None):
    """
    Plot samples from a 7D Gaussian distribution.
    
    Parameters:
        mean (numpy.ndarray): The mean of the Gaussian.
        cov (numpy.ndarray): The covariance matrix of the Gaussian.
        num_samples (int): The number of samples to generate.
    """
    labels = ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']


    # Generate samples
    samples = np.random.multivariate_normal(mean, cov, num_samples)
    print("Samples shape: ", samples.shape)
    
    # Plotting in 2D for visualization
    # fig, ax = plt.subplots(7, 7, figsize=(15, 15))
    # ax = ax.flatten()
    
    for i in range(7):
        for j in range(i+1, 7):
            # # visualize each distribution with a color map to denote density
            # ax[i*7 + j - 1].hist2d(samples[:, i], samples[:, j], bins=100, cmap='Blues', density=True)
            # ax[i*7 + j - 1].set_xlabel(labels[i])
            # ax[i*7 + j - 1].set_ylabel(labels[j])


            # ax[i*7 + j - 1].scatter(samples[:, i], samples[:, j], alpha=0.5)
            # ax[i*7 + j - 1].set_xlabel(labels[i])
            # ax[i*7 + j - 1].set_ylabel(labels[j])

            plt.hist2d(samples[:, i], samples[:, j], bins=100, cmap='Blues', density=True)
            plt.xlabel(labels[i])
            plt.ylabel(labels[j])
            # add color bar
            plt.colorbar()
            plt.show()

        #     for i in range(7):
        # for j in range(i+1, 7):
            # ax[i*7 + j - 1].hist2d(samples[:, i], samples[:, j], bins=100, cmap='Blues', density=True)
            # ax[i*7 + j - 1].set_xlabel(labels[i])
            # ax[i*7 + j - 1].set_ylabel(labels[j])


    # # Plot the original data if provided
    # if data is not None:
    #     for i in range(7):
    #         for j in range(i+1, 7):
    #             ax[i*7 + j - 1].scatter(data[:, i], data[:, j], color='red', alpha=0.5)
    
    # plt.tight_layout()
    # plt.show()