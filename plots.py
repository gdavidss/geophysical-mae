import torch
import numpy as np
import matplotlib.pyplot as plt

def plot_original_vs_reconstructed(original: torch.Tensor, masked: torch.Tensor, reconstructed: torch.Tensor, logger, n=4, filename="./mae-pretraining/original_v_reconstructed.png"):
    """
    Plots the original, masked, and reconstructed images of n samples with color bars.

    Args:
        original (torch.Tensor): The original images tensor of shape (batch_size, channels, height, width).
        masked (torch.Tensor): The masked images tensor of shape (batch_size, channels, height, width).
        reconstructed (torch.Tensor): The reconstructed images tensor of shape (batch_size, channels, height, width).
        n (int): Number of samples to plot.
        filename (str): The filename to save the plot.

    Note:
        Only the first channel (channel 1) is plotted for each sample, as imshow is used, which displays a single channel at a time.
    """
    fig, axes = plt.subplots(n, 3, figsize=(15, 15))
    for i in range(n):
        idx = np.random.randint(0, original.size(0))

        img1 = axes[i, 0].imshow(original[idx].cpu().numpy().squeeze(), cmap='viridis')  # Display the image with colormap
        axes[i, 0].set_title('Original')
        fig.colorbar(img1, ax=axes[i, 0])  # Add color bar to the original image

        img2 = axes[i, 1].imshow(masked[idx].cpu().numpy().squeeze(), cmap='viridis')  # Display the image with colormap
        axes[i, 1].set_title('Masked')
        fig.colorbar(img2, ax=axes[i, 1])  # Add color bar to the masked image

        img3 = axes[i, 2].imshow(reconstructed[idx].detach().cpu().numpy().squeeze(), cmap='viridis')  # Display the image with colormap
        axes[i, 2].set_title('Reconstructed')
        fig.colorbar(img3, ax=axes[i, 2])  # Add color bar to the reconstructed image

    plt.tight_layout()
    logger.experiment["original_v_reconstructed"].upload(fig)
    plt.savefig(filename)
    plt.close(fig)  # Close the figure after saving to free up memory

def vit_plot_original_vs_reconstructed(original: torch.Tensor, masked: torch.Tensor, reconstructed: torch.Tensor, n=4, filename="./mae-pretraining/original_v_reconstructed.png"):
    """
    Plots the original, masked, and reconstructed images of n samples with color bars.

    Args:
        original (torch.Tensor): The original images tensor of shape (batch_size, channels, height, width).
        masked (torch.Tensor): The masked images tensor of shape (batch_size, channels, height, width).
        reconstructed (torch.Tensor): The reconstructed images tensor of shape (batch_size, channels, height, width).
        n (int): Number of samples to plot.
        filename (str): The filename to save the plot.

    Note:
        Only the first channel (channel 1) is plotted for each sample, as imshow is used, which displays a single channel at a time.
    """
    fig, axes = plt.subplots(n, 3, figsize=(15, 15))

    vmin, vmax = None, None

    for i in range(n):
        idx = i  # np.random.randint(0, original.size(0))

        img1 = axes[i, 0].imshow(original[idx, 0].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)  # Display the first channel with colormap
        axes[i, 0].set_title('Original (Channel 1)')
        fig.colorbar(img1, ax=axes[i, 0])  # Add color bar to the original image

        img2 = axes[i, 1].imshow(masked[idx, 0].cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)  # Display the first channel with colormap
        axes[i, 1].set_title('Masked (Channel 1)')
        fig.colorbar(img2, ax=axes[i, 1])  # Add color bar to the masked image

        img3 = axes[i, 2].imshow(reconstructed[idx, 0].detach().cpu().numpy(), cmap='viridis', vmin=vmin, vmax=vmax)  # Display the first channel with colormap
        axes[i, 2].set_title('Reconstructed (Channel 1)')
        fig.colorbar(img3, ax=axes[i, 2])  # Add color bar to the reconstructed image


    plt.tight_layout()
    plt.close(fig)

    return fig


def plot_original_vs_masked(original: torch.Tensor, masked: torch.Tensor, n=4, filename="./mae-pretraining/original_v_masked.png"):
    """
    Plots the original and masked images of n samples with color bars.

    Args:
        original (torch.Tensor): The original images tensor of shape (batch_size, channels, height, width).
        masked (torch.Tensor): The masked images tensor of shape (batch_size, channels, height, width).
        n (int): Number of samples to plot.
        filename (str): The filename to save the plot.

    Note:
        Only the first channel (channel 1) is plotted for each sample, as imshow is used, which displays a single channel at a time.
    """
    fig, axes = plt.subplots(n, 2, figsize=(10, 15))
    for i in range(n):
        idx = np.random.randint(0, original.size(0))
        img1 = axes[i, 0].imshow(original[idx, 0].cpu().numpy(), cmap='viridis')  # Display the first channel with colormap
        axes[i, 0].set_title('Original (Channel 1)')
        fig.colorbar(img1, ax=axes[i, 0])  # Add color bar to the original image

        img2 = axes[i, 1].imshow(masked[idx, 0].cpu().numpy(), cmap='viridis')  # Display the first channel with colormap
        axes[i, 1].set_title('Masked (Channel 1)')
        fig.colorbar(img2, ax=axes[i, 1])  # Add color bar to the masked image

    plt.tight_layout()
    plt.savefig(filename)


def plot_loss_history(loss_history, filename="./mae-pretraining/loss_history.png"):
    plt.plot(range(1, len(loss_history) + 1), loss_history, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Loss (MSE)')
    plt.title('Pretraining Loss History')
    plt.yscale('log')
    plt.savefig(filename)
