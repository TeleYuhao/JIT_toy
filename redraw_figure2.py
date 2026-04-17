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

"""Redraw Figure 2 with KDE from saved model checkpoints.

This script reloads trained model checkpoints and generates KDE
visualization, useful for reproducing figures without
retraining.

Example:
    Run the redraw script:
        python3 redraw_figure2.py

    Outputs:
        - toy_results/figure_2_kde_v2.png
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import gaussian_kde

# Use non-interactive backend.
matplotlib.use('Agg')

# Set random seed.
np.random.seed(42)
torch.manual_seed(42)


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
        Generated samples on CPU, shape (num_samples, D).
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

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

    return z.cpu()


def plot_kde(
    ax,
    data_2d: np.ndarray,
    color: str,
    axis_lim: float = 2.5,
    is_gt: bool = False,
) -> None:
    """Plots 2D data with KDE contour visualization.

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

    # KDE rendering.
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

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect('equal')


def main():
    """Main function to redraw Figure 2 from saved models."""
    print('=' * 60)
    print('Redraw Figure 2 with KDE from saved models')
    print('=' * 60)

    # Auto-detect device.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

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

    # Load models and generate samples.
    for D in D_values:
        print(f'\nProcessing D = {D}...')

        D_results = []

        # Generate ground truth.
        x_gt = sample_ring(num_samples, radius=1.0, width=0.3)
        D_results.append(x_gt)

        for pred_type in prediction_types:
            model_file = f'./toy_results/model_D{D}_{pred_type}.pt'
            print(f'  Loading {model_file}...')

            # Load checkpoint.
            checkpoint = torch.load(model_file, map_location=device)

            # Reconstruct model and load weights.
            model = Generator(D + 1, 256, D).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])

            # Sample using Heun solver.
            z_final = sample_heun(
                model, D, pred_type,
                num_samples=num_samples,
                num_steps=50,
                device=device,
            )

            # Project back to 2D.
            P = checkpoint['P']
            if P.device.type == 'cuda':
                P = P.cpu()
            samples_d = (z_final.cpu() @ P).numpy()
            D_results.append(samples_d)

            r = np.sqrt(samples_d[:, 0] ** 2 + samples_d[:, 1] ** 2)
            print(f'    {pred_type}: radius={r.mean():.3f} +/- {r.std():.3f}')

        all_results.append(D_results)

    # Create KDE figure.
    print('\nCreating Figure 2 (KDE)...')

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    for di, D in enumerate(D_values):
        ax = axes[di, 0]
        plot_kde(ax, all_results[di][0], '#2ca02c', axis_lim=2.5, is_gt=True)
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
            plot_kde(ax, data, pred_colors[pred_type], axis_lim=2.5)
            if di == 0:
                ax.set_title(type_names[pred_type], fontsize=12, fontweight='bold')
            if di < 3:
                ax.set_xticklabels([])
            if di > 0:
                ax.set_yticklabels([])

    plt.savefig('./toy_results/figure_2_kde_v2.png', dpi=200, facecolor='white')
    plt.close()
    print('Saved: ./toy_results/figure_2_kde_v2.png')

    print('\nDone!')


if __name__ == '__main__':
    main()