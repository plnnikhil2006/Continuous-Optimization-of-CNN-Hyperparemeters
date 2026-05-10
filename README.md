# CNN Hyperparameter Optimization using Metaheuristics

**Optimization Techniques — Course Project**

A complete implementation of continuous hyperparameter optimization for a Convolutional Neural Network (CNN) trained on the MNIST dataset. The problem is solved using two gradient-free metaheuristic algorithms: Particle Swarm Optimization (PSO) and a Genetic Algorithm (GA).

---

## Table of Contents

1. [Problem Overview](#1-problem-overview)
2. [Mathematical Formulation](#2-mathematical-formulation)
3. [Environment Model](#3-environment-model)
4. [Algorithms](#4-algorithms)
5. [Project Structure](#5-project-structure)
6. [Installation](#6-installation)
7. [Usage](#7-usage)
8. [Output Files](#8-output-files)
9. [Results](#9-results)
10. [Discussion](#10-discussion)

---

## 1. Problem Overview

Given a standard Convolutional Neural Network architecture, select an optimal combination of continuous hyperparameters that simultaneously:

- Maximizes the predictive accuracy of the model on unseen data.
- Minimizes the validation loss.

Subject to hard constraints:
- Parameters must strictly reside within theoretically sound bounds (e.g., probability values cannot exceed 0.5 for dropout to maintain network stability).

This is a continuous optimization problem ($X_i \in \mathbb{R}^n$). Because the objective function involves training a neural network, the loss landscape is non-convex, noisy, and computationally expensive to evaluate, making metaheuristics the appropriate approach over traditional grid search or gradient descent.

---

## 2. Mathematical Formulation

### Decision Variables

Let the decision vector be `x = [alpha, p]^T`, where:
- `alpha`: Learning Rate, a continuous real number controlling the gradient descent step size.
- `p`: Dropout Rate, a continuous real number representing the probability of zeroing out a neuron's activation.

### Objective Function

**f(x) — Validation Loss (minimize):**

```text
min f(x) = L_val(w*; x)
```

Where `L_val` is the cross-entropy validation loss, and `w*` represents the optimal network weights obtained after training the CNN using the hyperparameters specified by `x`.

### Constraints

**Parameter Boundaries:**
```text
10^-4 <= alpha <= 10^-1
0.0 <= p <= 0.5
```
*Note: To effectively search the highly sensitive learning rate space, `alpha` is optimized in the log-10 domain (i.e., searching between -4.0 and -1.0).*

---

## 3. Environment Model

### Data Layout

To balance realistic training dynamics with the computational constraints of an optimization loop, a balanced subset of the MNIST dataset is utilized:
- **Training Set:** 5,000 samples
- **Validation Set:** 1,000 samples
- **Batch Size:** 128

### CNN Architecture

A custom `SimpleCNN` model built in PyTorch:
1. `Conv2d` (1 input channel, 16 output channels, kernel=3)
2. `ReLU` activation
3. `MaxPool2d` (kernel=2)
4. `Flatten`
5. `Linear` (Fully connected, 128 units)
6. `Dropout` (Rate determined by decision variable `p`)
7. `Linear` (Output layer, 10 classes)

The network is optimized using the Adam optimizer, with the learning rate determined by decision variable `alpha`.

---

## 4. Algorithms

### 4.1 Particle Swarm Optimization (PSO)

File: `src/main.py` (PSO implementation)

| Parameter | Value |
|-----------|-------|
| Swarm size (Particles) | 10 |
| Iterations | 10 |
| Inertia weight (w) | 0.5 |
| Cognitive constant (c1) | 1.5 |
| Social constant (c2) | 1.5 |
| Search Space | Continuous |

Key design choices:
- Particles update their velocity and position based on their personal best (`pbest`) and the swarm's global best (`gbest`).
- Strict boundary clamping (`np.clip`) is applied after every velocity update to ensure particles do not explore invalid dropout rates or unstable learning rates.

### 4.2 Genetic Algorithm (GA)

File: `src/main.py` (GA implementation)

| Parameter | Value |
|-----------|-------|
| Population size | 10 |
| Generations | 10 |
| Selection type | Tournament (k=2) |
| Crossover rate | 0.80 |
| Mutation rate | 0.20 |
| Crossover type | Arithmetic (Continuous blend) |
| Mutation type | Gaussian perturbation |

Key design choices:
- **Arithmetic Crossover:** Standard binary crossover fails for continuous variables. We use a linear combination of parent vectors: `child = beta * P1 + (1 - beta) * P2`.
- **Gaussian Mutation:** Adds a small random value drawn from `N(0, 0.1)` to maintain population diversity and prevent premature convergence.

---

## 5. Project Structure

```text
CNN_Hyperparameter_Optimization/
|
|-- report/                       # Documentation and LaTeX report
|   |-- report.tex              
|   `-- report.pdf              
|
|                  
|-- OT_Project.ipynb        # Interactive experimentation
|
|                      
|-- main.py            # Main source code    
|       
|-- optimization_results.png  # Generated output artifacts
|
|-- summary.md                  # High-level project summary
`-- README.md                   # This file
```

---

## 6. Installation

### Requirements

- Python 3.8 or higher
- pip
- CUDA toolkit (Optional but highly recommended for GPU acceleration)

### Install dependencies

```bash
pip install -r requirements.txt
```

Contents of `requirements.txt`:

```text
torch>=2.0.0
torchvision>=0.15.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

---

## 7. Usage

### Execution

The project is designed to be run from the command line. Ensure you are in the root directory of the repository.

**Run the full optimization suite:**
```bash
python src/main.py
```

The script automatically detects if a CUDA-enabled GPU is available and offloads the PyTorch tensors accordingly. Progress will be printed to the console for each PSO iteration and GA generation.

---

## 8. Output Files

### Figures (`results/`)

| File | Description |
|------|-------------|
| `optimization_results.png` | A dual-panel figure containing the Convergence Curve (Validation Loss over generations) and the Parameter Space Exploration scatter plot showing every evaluated combination of Learning Rate and Dropout. |

---

## 9. Results

Full run results with Population/Swarm size = 10, over 10 iterations/generations.

| Algorithm | Optimal Learning Rate | Optimal Dropout | Best Validation Loss |
|-----------|-----------------------|-----------------|----------------------|
| PSO       | 0.01197               | 0.296           | 0.1740               |
| GA        | 0.00853               | 0.122           | 0.1759               |

**PSO achieves the best overall fitness.** Both algorithms successfully navigated the non-convex loss landscape to find highly performant hyperparameters. 

### Parameter Exploration Findings

**Learning Rate Sensitivity:** The landscape is highly sensitive to the learning rate. Points evaluated outside the `10^-3` to `10^-2` range yielded massive spikes in validation loss, indicating failure to converge or gradient explosion.

**Dropout Rate Sensitivity:** The landscape is relatively robust to changes in dropout. Variations between 0.1 and 0.3 resulted in only marginal fluctuations in final model performance, confirming that the optimization landscape is highly anisotropic.

---

## 10. Discussion

### Why these algorithms suit this problem

Hyperparameter tuning of CNNs represents a continuous optimization problem where the objective function lacks analytical derivatives. Traditional gradient descent cannot optimize architectural choices like dropout. 

- **PSO** proved highly efficient in this low-dimensional continuous space. It rapidly clustered around the optimal learning rate and iteratively refined the global best.
- **GA** maintained higher diversity across the search space due to its Gaussian mutation. While it explored more of the domain, its convergence was slightly slower than PSO within the constrained iteration budget.

### Local optima analysis

Due to the limited computational budget (10 iterations), getting trapped in local optima is a significant threat. 
- The **GA** exhibited a tendency to stagnate around generation 5. Because it relies heavily on random mutation to escape local basins, the small population size prevented it from finding deeper minima.
- **PSO** successfully escaped early stagnation. It updated its global best effectively, pulling the swarm into a deeper basin of attraction by iteration 8.

### Constraint satisfaction

Boundary constraints were strictly maintained. In PSO, positions were clipped immediately after velocity updates. In GA, bounds were enforced post-mutation. This ensured that PyTorch never received invalid parameters (e.g., negative learning rates), preventing runtime crashes.

---

## Dependencies

| Package | Purpose |
|---------|---------|
| torch | Deep learning backend, tensor computation, autograd |
| torchvision | MNIST dataset loading and image transformations |
| numpy | Vectorized math for PSO/GA arrays and bound enforcement |
| matplotlib | Generating convergence and scatter plots |
| seaborn | Aesthetic styling for generated plots |

---

## Known Limitations

- Optimization is performed on a subset of the MNIST dataset for only 2 epochs per evaluation. While this proves the concept, the optimal parameters might shift if trained on the full dataset until complete convergence.
- The dimensionality of the problem is relatively low (2 variables).
- Fixed architecture size. The metaheuristics do not optimize the number of layers or filters.

---

## License

This project is developed for educational purposes as part of an Optimization Techniques course project.
