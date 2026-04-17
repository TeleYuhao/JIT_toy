# Copyright 2024 The Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Toy Experiment: Reproducing Figure 2 of "Back to Basics".

This script reproduces Figure 2 from the paper "Back to Basics: Let Denoising
Generative Models Denoise" (arXiv 2511.13720v2).

The experiment demonstrates that when observation dimension D >> data manifold
dimension d, only x-prediction succeeds while epsilon-prediction and
v-prediction fail when the model is under-complete.

Example:
    Run the experiment:
        python3 create_figure2.py

    Outputs:
        - toy_results/figure_2_kde.png
        - toy_results/figure_2_scatter.png
        - toy_results/model_D{D}_{pred_type}.pt (12 model files)
"""

import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility.
torch.manual_seed(42)
np.random.seed(42)

# Use non-interactive backend for matplotlib.
matplotlib.use('Agg')
warnings.filterwarnings('ignore')


class Generator(nn.Module):
    """MLP generator for the toy experiment.

    A 5-layer ReLU MLP that predicts x, epsilon, or velocity
    given noisy input z_t and time t.

    Attributes:
        net: Sequential MLP network.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        """Initializes the generator.

        Args:
            input_dim: Input dimension (D + 1 for z_t and t).
            hidden_dim: Hidden layer dimension.
            output_dim: Output dimension (D).
        """
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * 5 + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward pass of the generator.

        Args:
            z_t: Noisy data at time t, shape (batch_size, D).
            t: Time values, shape (batch_size,) or (batch_size, 1).

        Returns:
            Network output, shape (batch_size, D).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([z_t, t], dim=-1)
        return self.net(x)


def sample_ring(n: int, radius: float = 1.0, width: float = 0.3) -> np.ndarray:
    """Samples points from a ring distribution.

    Generates n points uniformly distributed on a ring with given
    radius and width.

    Args:
        n: Number of samples.
        radius: Radius of the ring.
        width: Width of the ring band.

    Returns:
        Array of shape (n, 2) with (x, y) coordinates.
    """
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = radius + np.random.uniform(-width, width, n)
    x = r * np.cos(theta)
    y = np.sin(theta) * r
    return np.column_stack([x, y])


def train_model(
    D: int,
    d: int,
    pred_type: str,
    hidden_dim: int = 256,
    epochs: int = 5000,
    batch_size: int = 512,
    lr: float = 2e-4,
    device: torch.device = None,
    save_dir: str = './toy_results',
) -> tuple[nn.Module, torch.Tensor]:
    """Trains a generator model for one prediction type.

    Trains a 5-layer ReLU MLP with the specified prediction target
    (x, epsilon, or velocity) on D-dimensional data.

    Args:
        D: Observation dimension.
        d: Data manifold dimension.
        pred_type: Prediction type ('x', 'eps', or 'v').
        hidden_dim: Hidden layer dimension (default: 256).
        epochs: Number of training epochs (default: 5000).
        batch_size: Batch size (default: 512).
        lr: Learning rate (default: 2e-4).
        device: Device to train on (default: auto-detect).
        save_dir: Directory to save model checkpoints.

    Returns:
        Tuple of (trained model, projection matrix P).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create random column-orthogonal projection matrix P.
    P = torch.randn(D, d, device=device)
    P, _ = torch.linalg.qr(P)

    # Initialize model and optimizer.
    model = Generator(D + 1, hidden_dim, D).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f'  Training {pred_type} for D={D} on {device}...')

    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    best_model_state = None

    for epoch in range(epochs):
        model.train()

        # Sample time and data.
        t = torch.rand(batch_size, device=device).clamp(0.01, 0.99)
        x_hat = torch.from_numpy(sample_ring(batch_size, radius=1.0, width=0.3)).float().to(device)
        x = x_hat @ P.T
        eps = torch.randn(batch_size, D, device=device)
        z_t = t.view(-1, 1) * x + (1 - t.view(-1, 1)) * eps

        # Compute target velocity.
        v_target = x - eps
        output = model(z_t, t)

        # Convert prediction to velocity based on prediction type.
        if pred_type == 'x':
            t_clamp = t.view(-1, 1).clamp(0, 0.999)
            v_pred = (output - z_t) / (1 - t_clamp)
        elif pred_type == 'eps':
            v_pred = (z_t - output) / t.view(-1, 1).clamp(min=0.01)
        elif pred_type == 'v':
            v_pred = output

        # Compute v-loss and backpropagate.
        loss = torch.mean((v_pred - v_target) ** 2)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Save best model.
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict().copy()

        if (epoch + 1) % 1000 == 0:
            print(f'    Epoch {epoch + 1}: loss={loss.item():.4f}')

    # Save best model checkpoint.
    save_path = os.path.join(save_dir, f'model_D{D}_{pred_type}.pt')
    torch.save({
        'model_state_dict': best_model_state,
        'P': P.cpu(),
        'D': D,
        'd': d,
        'pred_type': pred_type,
        'best_loss': best_loss,
    }, save_path)
    print(f'    Saved model: {save_path}')

    model.load_state_dict(best_model_state)
    return model, P


def sample_heun(
    model: nn.Module,
    D: int,
    pred_type: str,
    num_samples: int = 2000,
    num_steps: int = 50,
    device: torch.device = None,
) -> torch.Tensor:
    """Samples from the trained model using Heun solver.

    Uses 2nd-order Heun method to solve the ODE for sampling.

    Args:
        model: Trained generator model.
        D: Observation dimension.
        pred_type: Prediction type ('x', 'eps', or 'v').
        num_samples: Number of samples to generate.
        num_steps: Number of sampling steps.
        device: Device for sampling.

    Returns:
        Generated samples of shape (num_samples, D).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, D, device=device)
        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cur = step / num_steps
            if t_cur >= 1.0:
                break
            t_next = t_cur + dt

            # First Heun stage.
            t_batch = torch.full((num_samples,), t_cur, device=device)
            output1 = model(z, t_batch)

            if pred_type == 'x':
                x_pred1 = output1
            elif pred_type == 'eps':
                t_c = max(t_cur, 0.001)
                x_pred1 = (z - (1 - t_c) * output1) / t_c
            elif pred_type == 'v':
                x_pred1 = z + (1 - t_cur) * output1

            eps_pred1 = (z - t_cur * x_pred1) / max(1 - t_cur, 0.001)
            v_pred1 = x_pred1 - eps_pred1

            z_next = z + dt * v_pred1

            # Second Heun stage.
            t_next_c = min(t_next, 0.999)
            t_batch2 = torch.full((num_samples,), t_next_c, device=device)
            output2 = model(z_next, t_batch2)

            if pred_type == 'x':
                x_pred2 = output2
            elif pred_type == 'eps':
                t_n = max(t_next_c, 0.001)
                x_pred2 = (z_next - (1 - t_n) * output2) / t_n
            elif pred_type == 'v':
                x_pred2 = z_next + (1 - t_next_c) * output2

            eps_pred2 = (z_next - t_next_c * x_pred2) / max(1 - t_next_c, 0.001)
            v_pred2 = x_pred2 - eps_pred2

            # Combine Heun stages.
            z = z + 0.5 * dt * (v_pred1 + v_pred2)

    return z


def plot_kde_or_scatter(
    ax,
    data_2d: np.ndarray,
    color: str,
    axis_lim: float = 2.5,
    is_gt: bool = False,
) -> None:
    """Plots 2D data with KDE or scatter fallback.

    Args:
        ax: Matplotlib axes.
        data_2d: Array of shape (n, 2) with (x, y) coordinates.
        color: Plot color.
        axis_lim: Axis limit.
        is_gt: Whether this is ground truth (uses green colormap).
    """
    dist = np.sqrt(data_2d[:, 0] ** 2 + data_2d[:, 1] ** 2)
    mask = dist < axis_lim * 1.5
    data_2d = data_2d[mask]

    if len(data_2d) < 10:
        ax.text(0.5, 0.5, 'NaN', transform=ax.transAxes, ha='center', va='center',
               fontsize=14, color='gray')
        return

    # Try KDE rendering.
    try:
        from scipy.stats import gaussian_kde

        xy = np.vstack([data_2d[:, 0], data_2d[:, 1]])
        kde = gaussian_kde(xy, bw_method=0.15)

        x_grid = np.linspace(-axis_lim, axis_lim, 200)
        y_grid = np.linspace(-axis_lim, axis_lim, 200)
        X, Y = np.meshgrid(x_grid, y_grid)
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde(positions), X.shape)

        ax.contourf(X, Y, Z, levels=20, cmap='Greens' if is_gt else 'Blues',
                   alpha=0.85)
        ax.contour(X, Y, Z, levels=8, colors='#2ca02c' if is_gt else color,
                 linewidths=0.5, alpha=0.6)
    except Exception:  # Fallback to scatter.
        ax.scatter(data_2d[:, 0], data_2d[:, 1],
                 s=8, alpha=0.6, c=color, edgecolors='none')

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect('equal')


def main():
    """Main function to run the toy experiment."""
    print('=' * 60)
    print('Figure 2 - Matching Paper Setup')
    print('=' * 60)

    # Auto-detect device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    os.makedirs('./toy_results', exist_ok=True)

    # Experiment parameters.
    d = 2
    D_values = [2, 8, 16, 512]
    num_samples = 2000
    prediction_types = ['x', 'eps', 'v']
    type_names = {'x': 'x-pred', 'eps': 'eps-pred', 'v': 'v-pred'}
    pred_colors = {
        'x': '#1f77b4',
        'eps': '#ff7f0e',
        'v': '#9467bd',
    }

    all_results = []
    all_P = []
    np.random.seed(42)

    # Train models for each D and prediction type.
    for D in D_values:
        print(f'\nProcessing D = {D}...')

        D_results = []
        D_P = []

        # Generate ground truth.
        x_gt = sample_ring(num_samples, radius=1.0, width=0.3)
        D_results.append(x_gt)
        D_P.append(None)

        # Train and sample for each prediction type.
        for pred_type in prediction_types:
            model, P = train_model(D, d, pred_type, epochs=5000, device=device)
            z_final = sample_heun(
                model, D, pred_type,
                num_samples=num_samples,
                num_steps=50,
                device=device,
            )
            samples_d = (z_final @ P).cpu().numpy()
            D_results.append(samples_d)
            D_P.append(P)

            r = np.sqrt(samples_d[:, 0] ** 2 + samples_d[:, 1] ** 2)
            print(f'    {pred_type}: radius={r.mean():.3f} +/- {r.std():.3f}, '
                  f'in [-2.5,2.5]: {((r > 0) & (r < 2.5)).mean():.2%}')

        all_results.append(D_results)
        all_P.append(D_P)

    # Create KDE figure.
    print('\nCreating Figure 2 (KDE)...')

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for di, D in enumerate(D_values):
        ax = axes[di, 0]
        plot_kde_or_scatter(ax, all_results[di][0], '#2ca02c', axis_lim=2.5, is_gt=True)
        if di == 0:
            ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'D={D}', fontsize=12, fontweight='bold', rotation=0, labelpad=30)
        if di < 3:
            ax.set_xticklabels([])
        if di > 0:
            ax.set_yticklabels([])

        for pi, pred_type in enumerate(prediction_types):
            ax = axes[di, pi + 1]
            data = all_results[di][pi + 1]
            plot_kde_or_scatter(ax, data, pred_colors[pred_type], axis_lim=2.5)
            if di == 0:
                ax.set_title(type_names[pred_type], fontsize=12, fontweight='bold')
            if di < 3:
                ax.set_xticklabels([])
            if di > 0:
                ax.set_yticklabels([])

    plt.savefig('./toy_results/figure_2_kde.png', dpi=200, facecolor='white')
    plt.close()
    print('Saved: ./toy_results/figure_2_kde.png')

    # Create scatter figure.
    print('\nCreating Figure 2 (scatter)...')

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for di, D in enumerate(D_values):
        ax = axes[di, 0]
        ax.scatter(all_results[di][0][:, 0], all_results[di][0][:, 1],
                  s=3, alpha=0.5, c='#2ca02c', edgecolors='none')
        ax.set_xlim(-2.5, 2.5)
        ax.set_ylim(-2.5, 2.5)
        ax.set_aspect('equal')
        if di == 0:
            ax.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax.set_ylabel(f'D={D}', fontsize=12, fontweight='bold', rotation=0, labelpad=30)
        if di < 3:
            ax.set_xticklabels([])
        if di > 0:
            ax.set_yticklabels([])

        for pi, pred_type in enumerate(prediction_types):
            ax = axes[di, pi + 1]
            data = all_results[di][pi + 1]

            dist = np.sqrt(data[:, 0] ** 2 + data[:, 1] ** 2)
            mask = (dist > 0) & (dist < 5)
            data = data[mask]

            ax.scatter(data[:, 0], data[:, 1],
                     s=3, alpha=0.5, c=pred_colors[pred_type], edgecolors='none')
            ax.set_xlim(-2.5, 2.5)
            ax.set_ylim(-2.5, 2.5)
            ax.set_aspect('equal')
            if di == 0:
                ax.set_title(type_names[pred_type], fontsize=12, fontweight='bold')
            if di < 3:
                ax.set_xticklabels([])
            if di > 0:
                ax.set_yticklabels([])

    plt.savefig('./toy_results/figure_2_scatter.png', dpi=200, facecolor='white')
    plt.close()
    print('Saved: ./toy_results/figure_2_scatter.png')

    print('\nDone!')


if __name__ == '__main__':
    main()