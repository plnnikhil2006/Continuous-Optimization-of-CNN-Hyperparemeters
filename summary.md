# Project Summary

## Objective
This project applies Particle Swarm Optimization (PSO) and a Genetic Algorithm (GA) to perform hyperparameter tuning of a Convolutional Neural Network (CNN). [cite_start]The problem satisfies the "Problem A: Continuous Optimization" criteria by navigating a continuous real-numbered decision space[cite: 124, 125, 174].

## Methodology
The target was to minimize the objective function (validation loss of a CNN on the MNIST subset). The two continuous decision variables optimized were:
* Learning Rate (bounded logarithmically between 1e-4 and 1e-1)
* Dropout Rate (bounded linearly between 0.0 and 0.5)

## Results Comparison
Both algorithms were run on a PyTorch backend utilizing Google Colab's T4 GPU. 

| Algorithm | Optimal Learning Rate | Optimal Dropout Rate | Best Validation Loss |
|-----------|-----------------------|----------------------|----------------------|
| PSO       | 0.01197               | 0.296                | 0.1740               |
| GA        | 0.00853               | 0.122                | 0.1759               |

## Conclusion
Both metaheuristic algorithms successfully identified effective combinations of learning rate and dropout rate, significantly reducing the validation loss of the CNN model. PSO showed a slight edge in performance, achieving a marginally better validation loss (0.1740) compared to GA (0.1759). The approach proves that gradient-free optimization methods are highly capable of navigating complex, continuous deep learning search spaces where analytical derivatives are unavailable.
