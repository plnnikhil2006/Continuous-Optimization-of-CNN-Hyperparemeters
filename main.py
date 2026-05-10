import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import copy
import time

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
sns.set_theme(style="whitegrid")

# Configure Device for GPU execution
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# 1. Data Preparation (MNIST)
# ==========================================
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# Subset constraints for faster optimization loops
train_subset = Subset(train_dataset, range(5000))
val_subset = Subset(val_dataset, range(1000))

train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

# ==========================================
# 2. Model Architecture & Objective Function
# ==========================================
class SimpleCNN(nn.Module):
    def __init__(self, dropout_rate):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 14 * 14, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = x.view(-1, 16 * 14 * 14)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def evaluate_cnn(params, epochs=2):
    """
    Objective function: Minimizes Validation Loss.
    params[0]: Learning Rate Exponent (e.g., -3 for 1e-3)
    params[1]: Dropout Rate
    """
    lr = 10 ** params[0]
    dropout_rate = max(0.0, min(0.5, params[1]))

    model = SimpleCNN(dropout_rate).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    return val_loss / len(val_loader)

# ==========================================
# 3. Particle Swarm Optimization (PSO)
# ==========================================
def run_pso(num_particles=10, iterations=10):
    print("--- Starting PSO ---")
    start_time = time.time()
    bounds = np.array([[-4.0, -1.0], [0.0, 0.5]])
    particles = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(num_particles, 2))
    velocities = np.zeros((num_particles, 2))

    pbest = copy.deepcopy(particles)
    pbest_scores = np.array([float('inf')] * num_particles)
    gbest, gbest_score = None, float('inf')

    w, c1, c2 = 0.5, 1.5, 1.5
    convergence_curve = []
    history = []

    for i in range(iterations):
        for j in range(num_particles):
            score = evaluate_cnn(particles[j])
            history.append((particles[j][0], particles[j][1], score))

            if score < pbest_scores[j]:
                pbest_scores[j] = score
                pbest[j] = copy.deepcopy(particles[j])
            if score < gbest_score:
                gbest_score = score
                gbest = copy.deepcopy(particles[j])

        for j in range(num_particles):
            r1, r2 = np.random.rand(2)
            velocities[j] = (w * velocities[j] +
                             c1 * r1 * (pbest[j] - particles[j]) +
                             c2 * r2 * (gbest - particles[j]))
            particles[j] = np.clip(particles[j] + velocities[j], bounds[:, 0], bounds[:, 1])

        print(f"PSO Iteration {i+1}/{iterations} | Best Val Loss: {gbest_score:.4f}")
        convergence_curve.append(gbest_score)

    print(f"PSO Completed in {(time.time() - start_time):.2f} seconds.")
    return gbest, gbest_score, convergence_curve, history

# ==========================================
# 4. Genetic Algorithm (GA)
# ==========================================
def run_ga(pop_size=10, generations=10):
    print("\n--- Starting GA ---")
    start_time = time.time()
    bounds = np.array([[-4.0, -1.0], [0.0, 0.5]])
    population = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, 2))

    gbest, gbest_score = None, float('inf')
    convergence_curve = []
    history = []

    for gen in range(generations):
        scores = np.array([evaluate_cnn(ind) for ind in population])
        for idx, ind in enumerate(population):
            history.append((ind[0], ind[1], scores[idx]))

        min_idx = np.argmin(scores)
        if scores[min_idx] < gbest_score:
            gbest_score = scores[min_idx]
            gbest = copy.deepcopy(population[min_idx])

        parents = []
        for _ in range(pop_size):
            i, j = np.random.randint(0, pop_size, 2)
            parents.append(population[i] if scores[i] < scores[j] else population[j])
        parents = np.array(parents)

        next_gen = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i+1)%pop_size]
            if np.random.rand() < 0.8:
                alpha = np.random.rand()
                c1 = alpha * p1 + (1 - alpha) * p2
                c2 = (1 - alpha) * p1 + alpha * p2
            else:
                c1, c2 = p1, p2
            next_gen.extend([c1, c2])
        population = np.array(next_gen)[:pop_size]

        for i in range(pop_size):
            if np.random.rand() < 0.2:
                mutation = np.random.normal(0, 0.1, 2)
                population[i] = np.clip(population[i] + mutation, bounds[:, 0], bounds[:, 1])

        print(f"GA Generation {gen+1}/{generations} | Best Val Loss: {gbest_score:.4f}")
        convergence_curve.append(gbest_score)

    print(f"GA Completed in {(time.time() - start_time):.2f} seconds.")
    return gbest, gbest_score, convergence_curve, history

# ==========================================
# 5. Execution and Visualization
# ==========================================
if __name__ == "__main__":
    particles_pop = 10
    iters_gens = 10

    pso_best, pso_score, pso_curve, pso_history = run_pso(num_particles=particles_pop, iterations=iters_gens)
    ga_best, ga_score, ga_curve, ga_history = run_ga(pop_size=particles_pop, generations=iters_gens)

    print("\n=== FINAL OPTIMIZED PARAMETERS ===")
    print(f"PSO: LR = {10**pso_best[0]:.5f}, Dropout = {pso_best[1]:.3f} (Loss: {pso_score:.4f})")
    print(f"GA : LR = {10**ga_best[0]:.5f}, Dropout = {ga_best[1]:.3f} (Loss: {ga_score:.4f})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.plot(range(1, len(pso_curve)+1), pso_curve, marker='o', linewidth=2, label='PSO')
    ax1.plot(range(1, len(ga_curve)+1), ga_curve, marker='s', linewidth=2, label='GA')
    ax1.set_title('Optimization Convergence (Validation Loss)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Iteration / Generation', fontsize=12)
    ax1.set_ylabel('Validation Loss', fontsize=12)
    ax1.legend()

    pso_lrs, pso_drops, _ = zip(*pso_history)
    ga_lrs, ga_drops, _ = zip(*ga_history)

    ax2.scatter(pso_lrs, pso_drops, alpha=0.6, label='PSO Explored Points', marker='o')
    ax2.scatter(ga_lrs, ga_drops, alpha=0.6, label='GA Explored Points', marker='s')
    ax2.scatter(pso_best[0], pso_best[1], color='blue', s=200, edgecolors='black', label='PSO Optimal', zorder=5)
    ax2.scatter(ga_best[0], ga_best[1], color='orange', s=200, edgecolors='black', label='GA Optimal', zorder=5)

    ax2.set_title('Continuous Parameter Space Exploration', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Log10(Learning Rate)', fontsize=12)
    ax2.set_ylabel('Dropout Rate', fontsize=12)
    ax2.set_xlim([-4.1, -0.9])
    ax2.set_ylim([-0.05, 0.55])
    ax2.legend()

    plt.tight_layout()
    plt.savefig('optimization_results.png', dpi=300)
    plt.show()
