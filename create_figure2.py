import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os

torch.manual_seed(42)
np.random.seed(42)


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


def sample_ring(n, radius=1.0, width=0.3):
    theta = np.random.uniform(0, 2 * np.pi, n)
    r = radius + np.random.uniform(-width, width, n)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y])


def train_model(D, d, pred_type, hidden_dim=256, epochs=5000, batch_size=512, lr=2e-4):
    P = torch.randn(D, d)
    P, _ = torch.linalg.qr(P)

    model = Generator(D + 1, hidden_dim, D)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"  Training {pred_type} for D={D}...")

    for epoch in range(epochs):
        model.train()

        t = torch.rand(batch_size).clamp(0.01, 0.99)

        x_hat = torch.from_numpy(sample_ring(batch_size, radius=1.0, width=0.3)).float()
        x = x_hat @ P.T
        eps = torch.randn(batch_size, D)
        z_t = t.view(-1, 1) * x + (1 - t.view(-1, 1)) * eps

        v_target = x - eps
        output = model(z_t, t)

        if pred_type == 'x':
            t_clamp = t.view(-1, 1).clamp(0, 0.999)
            v_pred = (output - z_t) / (1 - t_clamp)
        elif pred_type == 'eps':
            v_pred = (z_t - output) / t.view(-1, 1).clamp(min=0.01)
        elif pred_type == 'v':
            v_pred = output

        loss = torch.mean((v_pred - v_target) ** 2)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 1000 == 0:
            print(f"    Epoch {epoch + 1}: loss={loss.item():.4f}")

    return model, P


def sample_heun(model, D, pred_type, num_samples=2000, num_steps=50):
    model.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, D)

        dt = 1.0 / num_steps

        for step in range(num_steps):
            t_cur = step / num_steps
            if t_cur >= 1.0:
                break
            t_next = t_cur + dt

            t_batch = torch.full((num_samples,), t_cur)
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
            t_batch2 = torch.full((num_samples,), t_next_c)
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

    return z


def plot_kde_or_scatter(ax, data_2d, color, axis_lim=2.5, is_gt=False):
    dist = np.sqrt(data_2d[:, 0] ** 2 + data_2d[:, 1] ** 2)
    mask = dist < axis_lim * 1.5
    data_2d = data_2d[mask]

    if len(data_2d) < 10:
        ax.text(0.5, 0.5, 'NaN', transform=ax.transAxes, ha='center', va='center',
                fontsize=14, color='gray')
        return

    try:
        from scipy.stats import gaussian_kde
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
    except Exception:
        ax.scatter(data_2d[:, 0], data_2d[:, 1],
                   s=3, alpha=0.5, c=color, edgecolors='none')

    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect('equal')


def main():
    print("=" * 60)
    print("Figure 2 - Matching Paper Setup")
    print("=" * 60)

    os.makedirs('./toy_results', exist_ok=True)

    d = 2
    D_values = [2, 8, 16, 512]
    num_samples = 2000
    prediction_types = ['x', 'eps', 'v']
    type_names = {'x': 'x-pred', 'eps': 'eps-pred', 'v': 'v-pred'}
    pred_colors = {'x': '#1f77b4', 'eps': '#ff7f0e', 'v': '#9467bd'}

    all_results = []
    all_P = []
    np.random.seed(42)

    for D in D_values:
        print(f"\nProcessing D = {D}...")

        D_results = []
        D_P = []

        x_gt = sample_ring(num_samples, radius=1.0, width=0.3)
        D_results.append(x_gt)
        D_P.append(None)

        for pred_type in prediction_types:
            model, P = train_model(D, d, pred_type, epochs=5000)
            z_final = sample_heun(model, D, pred_type, num_samples=num_samples, num_steps=50)
            samples_d = (z_final @ P).numpy()
            D_results.append(samples_d)
            D_P.append(P)

            r = np.sqrt(samples_d[:, 0] ** 2 + samples_d[:, 1] ** 2)
            print(f"    {pred_type}: radius={r.mean():.3f} +/- {r.std():.3f}, "
                  f"in [-2.5,2.5]: {((r > 0) & (r < 2.5)).mean():.2%}")

        all_results.append(D_results)
        all_P.append(D_P)

    print("\nCreating Figure 2 (KDE)...")

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

    plt.tight_layout(pad=0.8)
    plt.savefig('./toy_results/figure_2_kde.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: ./toy_results/figure_2_kde.png")

    print("\nCreating Figure 2 (scatter)...")

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

    plt.tight_layout(pad=0.8)
    plt.savefig('./toy_results/figure_2_scatter.png', dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print("Saved: ./toy_results/figure_2_scatter.png")

    print("\nDone!")


if __name__ == '__main__':
    main()
