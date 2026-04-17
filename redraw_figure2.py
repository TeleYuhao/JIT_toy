import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import os
import glob

class Generator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        layers = []
        dims = [input_dim] + [hidden_dim] * 5 + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([z_t, t], dim=-1)
        return self.net(x)


def sample_heun(model, D, pred_type, num_samples=2000, num_steps=50, device='cpu'):
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

            z = z + 0.5 * dt * (v_pred1 + v_pred2)

    return z.cpu()


def sample_ring(n, radius=1.0, width=0.3):
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = radius + np.random.uniform(-width, width, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def plot_kde(ax, data_2d, color, axis_lim=2.5, is_gt=False):
    dist = np.sqrt(data_2d[:, 0] ** 2 + data_2d[:, 1] ** 2)
    mask = dist < axis_lim * 1.5
    data_2d = data_2d[mask]

    if len(data_2d) < 10:
        ax.text(0.5, 0.5, 'NaN', transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray')
        return

    # Use KDE
    xy = np.vstack([data_2d[:, 0], data_2d[:, 1]])
    kde = gaussian_kde(xy, bw_method=0.15)

    x_grid = np.linspace(-axis_lim, axis_lim, 200)
    y_grid = np.linspace(-axis_lim, axis_lim, 200)
    X, Y = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(kde(positions), X.shape)

    ax.contourf(X, Y, Z, levels=20, cmap='Greens' if is_gt else 'Blues', alpha=0.85)
    ax.contour(X, Y, Z, levels=8, colors='#2ca02c' if is_gt else color,
               linewidths=0.5, alpha=0.6)

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect('equal')


def main():
    print("=" * 60)
    print("Redraw Figure 2 with KDE from saved models")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    d = 2
    D_values = [2, 8, 16, 512]
    num_samples = 2000
    prediction_types = ['x', 'eps', 'v']
    type_names = {'x': 'x-pred', 'eps': 'eps-pred', 'v': 'v-pred'}
    pred_colors = {'x': '#1f77b4', 'eps': '#ff7f0e', 'v': '#9467bd'}

    all_results = []

    for D in D_values:
        print(f"\nProcessing D = {D}...")

        D_results = []

        # Ground Truth
        x_gt = sample_ring(num_samples, radius=1.0, width=0.3)
        D_results.append(x_gt)

        for pred_type in prediction_types:
            model_file = f'./toy_results/model_D{D}_{pred_type}.pt'
            print(f"  Loading {model_file}...")
            
            checkpoint = torch.load(model_file, map_location=device)
            
            model = Generator(D + 1, 256, D).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            P = checkpoint['P'].to(device)

            z_final = sample_heun(model, D, pred_type, num_samples=num_samples, num_steps=50, device=device)
            P = checkpoint['P']
            if P.device.type == 'cuda':
                P = P.cpu()
            samples_d = (z_final.cpu() @ P).numpy()
            D_results.append(samples_d)

            r = np.sqrt(samples_d[:, 0] ** 2 + samples_d[:, 1] ** 2)
            print(f"    {pred_type}: radius={r.mean():.3f} +/- {r.std():.3f}")

        all_results.append(D_results)

    print("\nCreating Figure 2 (KDE)...")

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
    print("Saved: ./toy_results/figure_2_kde_v2.png")

    print("\nDone!")


if __name__ == '__main__':
    main()