# Inventory Optimization using Q-learning

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://opensource.org/licenses/MIT)

This project applies Q-learning; a popular reinforcement learning to optimize inventory management decisions under stochastic demand conditions. It benchmarks RL performance against a globally optimal (s,S) policy.

## 📌 Project Overview

### Question
"How well can Q-learning solve inventory management problems in stochastic environments?"

### Key Components
- **Inventory simulation environment** with stochastic demand
- **Q-learning agent** implementation
- **Global optimal policy search** for benchmarking
- **Performance evaluation** 

### Problem Formulation
**States**: Inventory Position inventory at end of day (0-60 vehicles). Inventory Position = on hand inventory + pending orders - back orders 
**Actions**: Order quantity (0-30 vehicles)  
**Reward**: -(Holding Cost + Ordering Cost + Shortage Cost)  
**Objective**: Minimize total costs over 100-day horizon

## 📊 Key Results
| Metric | Global Optimal | Q-learning |
|--------|----------------|------------|
| **Mean Total Cost** | $95,287 | $96,259 |
| **Shortage Probability** | 6 % | 6.9% |
| **Performance** | 100% | 99% |

Q_learning can learn 99% of global optimal cost.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Required packages:
```bash
pip install simpy numpy matplotlib seaborn
```

### Project Structure
```bash
inventory-rl/
├── result/                   # Simulation output data                 # Presentation slides
├── src/
│   ├── global_optimal_search.py     # Finds optimal (s,S) policy
│   ├── main.py                      # Trains & evaluates Q-agent
│   ├── Q_learning_agent.py          # Q-learning implementation
│   ├── simulate_inventory_policy.py # Simulate input inventory policy
│   └── simulation_model.py          # Inventory environment
└── README.md
```

### Running the Project

### Find global optimal policy:
```bash
python src/global_optimal_search.py
```

### Train and evaluate Q-learning agent:
```bash
python src/main.py
```
### More about the project 
Read : https://drive.google.com/file/d/1jEBPs7XjuP82YFKN3xL_6fNigLlqs4zz/view?usp=drive_link

### References 
1) Winston, W. L. (2004). Inventory theory. In Operations Research: Applications and Algorithms (4th ed., pp. [890-907]). Belmont, CA: Thomson/Brooks/Cole.
2) Sutton, R. S., & Barto, A. G. (2018). Temporal difference learning. In Reinforcement Learning: An Introduction (2nd ed., pp. 119–136). MIT Press.
3) Kara, A., & Dogan, I. (2018). Reinforcement learning approaches for specifying ordering policies of perishable inventory systems. Expert Systems with Applications, *91*, 150-158. https://doi.org/10.1016/j.eswa.2017.08.046
4) Rolf, B., Jackson, I., Müller, M., Lang, S., Reggelin, T., & Ivanov, D. (2023). A review on reinforcement learning algorithms and applications in supply chain management. International Journal of Production Research, *61*(20), 7151–7179. https://doi.org/10.1080/00207543.2022.2140221
5) Scarf, H. (1960). The optimality of (s, S) policies in the dynamic inventory problem. Reprinted 2019. Retrieved from http://dido.wss.yale.edu/~hes/pub/ss-policies.pdf
