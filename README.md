# CNN Hyperparameter Optimization using Metaheuristics

**Optimization Techniques — Final Course Project**

An end-to-end, rigorous implementation of continuous hyperparameter optimization for a Convolutional Neural Network (CNN) trained on the MNIST dataset. This repository addresses the "Problem A: Continuous Optimization" criteria by solving a highly non-convex, computationally expensive black-box optimization problem using two advanced gradient-free metaheuristic algorithms: Particle Swarm Optimization (PSO) and a Genetic Algorithm (GA).

---

## Table of Contents

1. Executive Summary and Problem Overview
2. Theoretical Background and Motivation
   2.1 The Curse of Dimensionality in Grid Search
   2.2 The Case for Metaheuristics
3. Mathematical Formulation
   3.1 Continuous Decision Variables
   3.2 Objective Function Definition
   3.3 Boundary Constraints and Feasibility
4. Environment and CNN Architecture
   4.1 Dataset Stratification and Subsampling
   4.2 Layer-by-Layer Network Topology
   4.3 Hardware and Computational Setup
5. Algorithmic Deep Dive: Metaheuristics
   5.1 Particle Swarm Optimization (PSO) Mathematics
   5.2 Genetic Algorithm (GA) Evolutionary Operators
6. Project Architecture and Code Structure
   6.1 Directory Tree
   6.2 Core Functional Modules
7. Installation and Environment Setup
8. Usage and Execution Instructions
9. Empirical Results and Performance
   9.1 Quantitative Comparison Table
   9.2 Convergence Trajectory Analysis
10. Deep Discussion and Landscape Analysis
    10.1 Parameter Sensitivity (Anisotropy)
    10.2 Navigating Local Optima Traps
    10.3 Constraint Enforcement Strategies
11. Known Limitations and Future Scope
12. License and Academic Declaration

---

## 1. Executive Summary and Problem Overview

Training a deep learning model involves two entirely distinct classes of parameters: 
First, there are Network Weights—the millions of internal parameters optimized automatically via backpropagation and gradient descent during the training loop. 
Second, there are Hyperparameters—the structural and operational settings (e.g., learning rate, dropout, layer counts, momentum) chosen by the engineer prior to training.

Selecting optimal hyperparameters is fundamentally a continuous optimization problem. However, the objective function (the validation loss of the network) lacks analytical derivatives with respect to these hyperparameters. You cannot simply calculate the gradient of the loss with respect to the dropout rate using calculus. Furthermore, the loss landscape of deep neural networks is notoriously non-convex, multimodal, and severely noisy. 

This project demonstrates how intelligent, population-based metaheuristics can systematically and efficiently navigate this complex search space to discover highly performant configurations with a strictly limited computational budget, vastly outperforming naive search methods.

---

## 2. Theoretical Background and Motivation

### 2.1 The Curse of Dimensionality in Grid Search
Historically, hyperparameter tuning was conducted using Grid Search, where a practitioner defines a discrete set of values for each parameter and tests every single combination. If we test 10 learning rates and 10 dropout rates, that requires 100 model trainings. If we add momentum, weight decay, and batch size, the combinations grow exponentially. This exponential explosion is known as the "curse of dimensionality." 

Random search performs slightly better by randomly sampling continuous distributions rather than a rigid grid, but it lacks "memory." It does not learn from previous poor evaluations to guide future searches, meaning it wastes massive amounts of computational power evaluating terrible parameter combinations.

### 2.2 The Case for Metaheuristics
Metaheuristics like Particle Swarm Optimization (PSO) and Genetic Algorithms (GA) offer a directed, intelligent search mechanism. They initialize a random population across the continuous search space, evaluate the objective function, and iteratively update their positions based on the success of past evaluations. They share information (swarm intelligence) or combine successful traits (crossover), converging on optimal basins of attraction much faster than exhaustive search. This makes them ideal for the computationally expensive task of training neural networks.

---

## 3. Mathematical Formulation

To strictly satisfy the criteria for a Continuous Optimization Problem, we must mathematically define the vectors, objectives, and constraints.

### 3.1 Continuous Decision Variables
The search space consists of a continuous, real-valued decision vector X = [x1, x2]^T, where:

* x1 (alpha - Learning Rate): A real number that controls the step size during the Adam optimization of the neural network weights. It dictates the speed and stability of the weight convergence. x1 is an element of all Real numbers.
* x2 (p - Dropout Rate): A structural regularization parameter representing the probability of randomly zeroing out a hidden neuron's activation during the forward pass to prevent complex co-adaptations and overfitting. x2 is an element of all Real numbers.

### 3.2 Objective Function Definition
The objective is treated as an expensive black-box evaluation function that we seek to minimize. The function involves initializing a CNN with the vector X, training it for a set number of epochs, and calculating the resultant error.

    Minimize f(X) = L_val(w*; X)

Where:
* f(X) is the scalar fitness value returned to the optimization algorithm.
* L_val is the Cross-Entropy validation loss computed on unseen data.
* w* represents the optimal network weights obtained after training the CNN using the hyperparameters mapped by the decision vector X.

### 3.3 Boundary Constraints and Feasibility
To ensure the metaheuristics search within theoretically sound regions and to prevent network collapse (e.g., negative learning rates causing mathematical exceptions, or 100% dropout destroying all information), strict boundary constraints are enforced:

* Learning Rate Bounds: 10^-4 <= x1 <= 10^-1
  Mathematical Note: Because neural networks are exponentially sensitive to learning rates, the search is executed in the continuous base-10 logarithmic domain, mapping the search space to [-4.0, -1.0]. The true learning rate is recovered via 10^x1 before being passed to PyTorch.

* Dropout Rate Bounds: 0.0 <= x2 <= 0.5
  Mathematical Note: Values exceeding 0.5 destroy too much information flow, causing the network to underfit severely. Negative values are theoretically impossible for probabilities.

---

## 4. Environment and CNN Architecture

### 4.1 Dataset Stratification and Subsampling
To balance realistic neural network training dynamics with the strict computational constraints of an iterative optimization loop (where the model is trained hundreds of times), a stratified subset of the standard MNIST dataset is utilized:
* Global Training Set: 60,000 images.
* Subsampled Training Set: 5,000 images (to speed up objective function evaluation).
* Validation Set: 1,000 images.
* Batch Size: 128 images per forward pass.

### 4.2 Layer-by-Layer Network Topology
A custom SimpleCNN model is built utilizing the PyTorch deep learning framework. The architectural topology is fixed to strictly isolate the effects of the continuous hyperparameters.

Layer Type   | Input Tensor Shape  | Output Tensor Shape | Parameters / Mechanics
------------ | ------------------- | ------------------- | ------------------------
Conv2d       | [128, 1, 28, 28]    | [128, 16, 28, 28]   | Kernel=3x3, Stride=1, Padding=1
ReLU         | [128, 16, 28, 28]   | [128, 16, 28, 28]   | Non-linear activation max(0, x)
MaxPool2d    | [128, 16, 28, 28]   | [128, 16, 14, 14]   | Kernel=2x2, Stride=2
Flatten      | [128, 16, 14, 14]   | [128, 3136]         | Reshapes tensor for dense layers
Linear       | [128, 3136]         | [128, 128]          | Fully connected dense layer
ReLU         | [128, 128]          | [128, 128]          | Non-linear activation
Dropout      | [128, 128]          | [128, 128]          | Rate p optimized by PSO/GA
Linear       | [128, 128]          | [128, 10]           | Output logits for 10 digits

### 4.3 Hardware and Computational Setup
The PyTorch backend is configured to automatically detect and bind to CUDA-enabled hardware (e.g., NVIDIA T4 GPU). If unavailable, it gracefully falls back to CPU execution. Tensor operations are parallelized to minimize the overhead of the objective function.

---

## 5. Algorithmic Deep Dive: Metaheuristics

This section details the mathematical mechanics of the two implemented optimization algorithms.

### 5.1 Particle Swarm Optimization (PSO) Mathematics
PSO is a swarm-intelligence metaheuristic inspired by the social foraging behavior of bird flocks. It maintains a swarm of particles, where each particle i has a position X_i and a velocity V_i.

Swarm Configuration:
* Swarm Size (N): 10 particles.
* Iterations (T): 10 update cycles.
* Inertia Weight (w): 0.5 (Provides momentum, preventing erratic direction changes).
* Cognitive Constant (c1): 1.5 (Pulls particle toward its historical personal best).
* Social Constant (c2): 1.5 (Pulls particle toward the swarm's global best).

Velocity Update Equation:
    V_i(t+1) = w * V_i(t) + c1 * r1 * (Pbest_i - X_i(t)) + c2 * r2 * (Gbest - X_i(t))

Where r1 and r2 are random vectors sampled uniformly from U(0,1) to provide stochastic exploration.

Position Update Equation:
    X_i(t+1) = X_i(t) + V_i(t+1)

Boundary Enforcement:
Velocity updates inherently risk pushing particles out of defined mathematical bounds. A strict numpy.clip(X_i, lower_bound, upper_bound) is applied immediately after every position matrix update.

### 5.2 Genetic Algorithm (GA) Evolutionary Operators
GA is an evolutionary metaheuristic mimicking natural selection, highly favored for maintaining diversity across multimodal landscapes.

Population Configuration:
* Population Size: 10 continuous chromosomes.
* Generations: 10 evolutionary cycles.
* Selection Mechanism: Tournament Selection (k=2). Two individuals are chosen at random; the one with the lower validation loss wins parent rights.

Continuous Space Adaptations:
1. Arithmetic Crossover (Rate = 0.80): Standard binary point-crossover is mathematically invalid for continuous real numbers. Instead, children are generated via a linear interpolation (blend) of parent vectors:
   
    Child_1 = beta * Parent_1 + (1 - beta) * Parent_2
    Child_2 = (1 - beta) * Parent_1 + beta * Parent_2
   
    Where beta is a random scalar from U(0,1).

2. Gaussian Mutation (Rate = 0.20):
    To inject fresh genetic material and prevent the population from converging prematurely on a local optimum, a scalar value drawn from a normal distribution N(0, 0.1) is added to the parameters.
   
    X_mutated = X + N(0, 0.1)
   
    Post-mutation, bounds are strictly enforced via clipping.

---

## 6. Project Architecture and Code Structure

### 6.1 Directory Tree
To maintain professional repository standards, the project is structured as follows:

    CNN_Hyperparameter_Optimization/
    |-- data/                         # Holds downloaded MNIST binaries.
    |-- docs/                         # Formal academic documentation.
    |   |-- report.tex                # LaTeX source code for the final report.
    |   |-- report.pdf                # Compiled, print-ready PDF report.
    |-- notebooks/                    # Prototyping and interactive environments.
    |   |-- OT_Project.ipynb          # Jupyter notebook for step-by-step visualization.
    |-- src/                          # Core execution codebase.
    |   |-- main.py                   # Consolidated execution script.
    |-- results/                      # Generated output artifacts and plots.
    |   |-- optimization_results.png  # Dual-panel visualization of algorithm convergence.
    |-- .gitignore                    # Excludes datasets, cache, and compiled LaTeX files.
    |-- requirements.txt              # Explicit list of Python environment dependencies.
    |-- summary.md                    # Brief high-level project summary for quick reading.
    |-- README.md                     # Comprehensive project documentation (this file).

### 6.2 Core Functional Modules (Inside src/main.py)
* class SimpleCNN(nn.Module): Defines the neural network topology.
* evaluate_cnn(params): The core objective function. Maps params[0] to log-learning rate and params[1] to dropout. Trains the network for 1 epoch, evaluates on the validation loader, and returns the scalar loss.
* run_pso(...): Encapsulates the particle swarm logic, maintaining pbest arrays, gbest scalars, and velocity tracking matrices.
* run_ga(...): Encapsulates the genetic algorithm, handling tournament selection, arithmetic blending, and stochastic mutation probabilities.
* __main__: The execution block that runs both optimizers, captures historical trajectories, and generates comparative Matplotlib plots.

---

## 7. Installation and Environment Setup

### 7.1 System Prerequisites
* Python: Version 3.8, 3.9, or 3.10.
* Package Manager: pip.
* Hardware: A multi-core CPU is sufficient, but a CUDA-toolkit compatible NVIDIA GPU (e.g., RTX 3060, Tesla T4) will reduce execution time by approximately 85%.

### 7.2 Environment Initialization

1. Clone the repository to your local machine:
    git clone https://github.com/YourUsername/CNN_Hyperparameter_Optimization.git
    cd CNN_Hyperparameter_Optimization

2. Isolate dependencies by creating a virtual environment:
    python -m venv venv

3. Activate the virtual environment:
    On Linux / macOS: source venv/bin/activate
    On Windows: venv\Scripts\activate.bat

4. Install all required dependencies:
    pip install -r requirements.txt

(Contents of requirements.txt includes: torch>=2.0.0, torchvision>=0.15.0, numpy>=1.21.0, matplotlib>=3.5.0, seaborn>=0.11.0)

---

## 8. Usage and Execution Instructions

The entire optimization pipeline is fully encapsulated within a single executable script to ensure reproducibility and ease of use. 

To execute the program, run the following command from the root directory:

    python src/main.py

Execution Flow and Expected Output:
1. Device Binding: The script will print 'Using device: cuda' or 'Using device: cpu'.
2. Dataset Verification: It will check the ./data directory. If MNIST is missing, it will automatically download the 60MB dataset from Yann LeCun's servers.
3. PSO Phase: Initializes 10 particles. Iteratively evaluates 100 total CNN models. Console logs the format: 'PSO Iteration X/10 | Best Val Loss: 0.XXXX'
4. GA Phase: Initializes 10 chromosomes. Evolves through 10 generations (100 total CNN evaluations). Console logs the format: 'GA Generation X/10 | Best Val Loss: 0.XXXX'
5. Plot Generation: A Matplotlib GUI window will appear displaying the convergence metrics. The plot is automatically saved to the results/ folder, and the final optimal parameters are printed to the console.

---


## 9. Empirical Results and Performance

### 9.1 Quantitative Comparison Table
The following results were obtained utilizing a Swarm/Population size of 10 across a maximum of 10 iterations, yielding exactly 100 objective function evaluations per algorithm.

Optimization Algorithm  | Optimal Learning Rate (Real) | Optimal Dropout Rate | Best Validation Loss | Convergence Speed
----------------------- | ---------------------------- | -------------------- | -------------------- | -----------------
Particle Swarm (PSO)    | 0.01197                      | 0.296                | 0.1740               | Iteration 8
Genetic Algorithm (GA)  | 0.00853                      | 0.122                | 0.1759               | Generation 5

### 9.2 Convergence Trajectory Analysis
Both metaheuristics successfully demonstrated the capacity to rapidly bypass unviable, highly-penalized parameter combinations (e.g., extremely high learning rates leading to gradient explosion) to locate high-performing regions in the continuous space. 

Particle Swarm Optimization:
PSO achieved the absolute best overall fitness, identifying a parameter vector that achieved a marginally superior validation loss. Its trajectory showed continuous improvement. The velocity mechanism allowed the swarm to rapidly collapse around the 0.01 learning rate magnitude.

Genetic Algorithm:
GA showed massive improvements in the first 3 generations as tournament selection rapidly killed off chromosomes with unstable learning rates. However, its convergence curve flattened out significantly after generation 5, struggling to perform the micro-adjustments required to beat PSO's final score.

---

## 10. Deep Discussion and Landscape Analysis

### 10.1 Parameter Sensitivity (Anisotropy)
By visualizing the historical evaluation coordinates on a scatter plot, we can analyze the geometry of the CNN loss landscape. The objective function is highly anisotropic (behaves differently along different axes).
* Learning Rate Axis (Steep and Sensitive): The objective landscape is incredibly steep along the alpha axis. Particles evaluating values outside the narrow 10^-3 to 10^-2 corridor experienced massive loss spikes. Any value approaching 10^-1 caused the Adam optimizer to overshoot minima entirely, failing to converge.
* Dropout Rate Axis (Flat and Robust): Conversely, the landscape proved highly robust to changes in the dropout dimension. Configurations varying widely between 0.10 and 0.35 resulted in near-identical validation metrics. The algorithms correctly identified that precise structural regularization was less critical than achieving the optimal optimization step-size.

### 10.2 Navigating Local Optima Traps
Given the restrictive evaluation budget (100 total objective function calls per algorithm), local optima posed a severe and ever-present threat. Deep learning landscapes are riddled with saddle points and shallow local minima.
* GA Stagnation: The Genetic Algorithm exhibited a clear tendency to plateau around Generation 5. Without a massive population (e.g., 100+) to sustain diverse genetic material, the tournament selection mechanism rapidly homogenized the gene pool. Once the population lost diversity, arithmetic crossover simply averaged similar numbers, trapping the algorithm in a shallow local optimum.
* PSO Momentum to the Rescue: PSO successfully escaped the early stagnation that trapped the GA. The inertia weight (w=0.5) provided mathematical momentum. Even when a particle reached a local minimum, its accumulated velocity forced it to overshoot the minimum slightly, allowing the swarm to traverse small loss ridges and settle into a deeper, superior basin of attraction by Iteration 8.

### 10.3 Constraint Enforcement Strategies
In unconstrained mathematical optimization, bounds are a suggestion. In software engineering and deep learning, bounds are a hard requirement. Feeding the PyTorch optimizer a negative learning rate causes a fatal ValueError runtime exception, immediately terminating the script.
* In PSO, constraints were enforced strictly via post-velocity positional clipping. If a velocity update pushed a particle's learning rate to -5.0 (log scale), it was hard-clamped back to -4.0 before evaluating the objective function.
* In GA, boundaries were enforced immediately following the Gaussian mutation step. This strict architectural boundary mapping guaranteed 100% functional feasibility and zero software crashes across all 200 model evaluations.

---

## 11. Known Limitations and Future Scope

While this project successfully fulfills the Continuous Optimization requirements, it operates under several controlled limitations:

1. Subsampling Bias: Evaluations are performed on a 5,000-sample subset for only 1 to 2 epochs per trial. While this is absolutely necessary to complete the optimization loop in a reasonable timeframe (under 10 minutes on a GPU), the discovered parameters act as proxies. If training the network on the complete 60,000-sample dataset until absolute convergence (e.g., 50 epochs), the optimal dropout rate might need to be higher to prevent long-term overfitting.

2. Dimensional Scope Constraints: The search space is intentionally limited to two continuous variables to satisfy the assignment scope and allow for clear 2D visualization of the search space. Real-world hyperparameter optimization often scales to 10+ dimensions, including momentum terms, weight decay (L2 penalty), batch sizes, and learning rate decay schedules.

3. Fixed Topology Architecture: The metaheuristics implemented here optimize continuous operational variables but do not engage in Neural Architecture Search (NAS). A future extension of this project could involve optimizing discrete structural variables (e.g., integer number of convolutional filters, integer kernel sizes, boolean inclusion of batch normalization layers).

4. Multi-Objective Expansion:
   Future iterations could implement NSGA-II (Non-dominated Sorting Genetic Algorithm II) to perform multi-objective optimization, attempting to simultaneously minimize Validation Loss while also minimizing Model Parameter Count (memory footprint) to find Pareto-optimal network designs.

---

## 12. License and Academic Declaration

Academic Integrity Declaration: This project, repository, and all associated code artifacts were developed as a formal submission for the Optimization Techniques course assignment. All continuous metaheuristic implementations (Particle Swarm Optimization and Genetic Algorithm) were written entirely from scratch specifically for this hyperparameter tuning problem, relying only on standard NumPy vectorization logic. No pre-packaged optimization libraries (like SciPy.optimize or Optuna) were used to bypass the algorithm implementation requirements.

License: This codebase is released under the MIT License for educational, academic, and non-commercial open-source purposes.
