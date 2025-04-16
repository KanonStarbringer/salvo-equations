# Salvo Equations Simulator

A comprehensive web application for naval combat modeling using various implementations of the Salvo Combat Model. This application provides interactive simulations of different naval combat scenarios, from basic salvo equations to complex Monte Carlo simulations.

## Features

- **Basic Salvo Model**: Classic implementation of the salvo equations
- **Continuous-Time Salvo**: Differential equation-based model for continuous combat simulation
- **Stochastic Salvo**: Probabilistic model with Gaussian noise in damage calculations
- **Multiple Forces Salvo**: Extension to handle multiple combatant forces
- **Monte Carlo Simulation**: Advanced model with multi-domain combat (air, naval, submarine)



## 📐 Mathematical Models

The Monte Carlo simulation used in this project is based on the following mathematical formulations:


% 1. Firepower Generation (Poisson-distributed shots per unit)
$$
S_{i}^{(d)} \sim \text{Poisson}(\lambda_{i}^{(d)})
$$

% 2. Probability of Hit and Interception
$$
H_{i \rightarrow j}^{(d_k \rightarrow d_l)} \sim \text{Binomial}\left(S_{i}^{(d_k)},\; p_{\text{hit},\,i}^{(d_k \rightarrow d_l)} \cdot (1 - p_{\text{int},\,j})\right)
$$

% 3. Unit Loss Update
$$
U_j^{(d_l)}(t+1) = \max\left(0,\; U_j^{(d_l)}(t) - \sum_{d_k} H_{i \rightarrow j}^{(d_k \rightarrow d_l)} \right)
$$

% 4. Reinforcement Probability (Adaptive)
$$
P_{\text{reinforce},\,i}^{(d)} = \min\left(0.02 + 0.1 \cdot \left(1 - \frac{\sum_d U_i^{(d)}(t)}{\sum_d U_i^{(d)}(0)}\right),\; 1.0\right)
$$

% 5. Morale Collapse Check
$$
\text{If } \frac{\sum_d U_i^{(d)}(t)}{\sum_d U_i^{(d)}(0)} < \theta_{\text{morale}}, \text{ then side } i \text{ collapses with probability } p_{\text{collapse}}
$$

% 6. Simulation Looping (Monte Carlo)
$$
\text{Repeat for each simulation } s = 1, 2, \dots, N_{\text{sim}} \text{ and each round } t = 1, 2, \dots, T_{\text{max}}
$$

These equations form the backbone of the probabilistic, dynamic, and multi-domain combat engine used in the simulator.


## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/salvo-equations-simulator.git
cd salvo-equations-simulator
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

To run the application:

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`.

## Project Structure

```
.
├── app.py              # Main Streamlit application
├── assets/            # Directory containing images and other static files
│   ├── CDDGN.png     # CDDGN logo
│   └── naval-battle.jpg # Background image
├── requirements.txt   # Python dependencies
└── README.md         # This file
```

## Models

### Basic Salvo Model
- Simulates combat between two forces
- Parameters include initial forces, firepower, and interception capabilities

### Continuous-Time Salvo
- Uses differential equations for continuous combat simulation
- Provides smooth force evolution over time

### Stochastic Salvo
- Adds Gaussian noise to damage calculations
- Accounts for uncertainty in combat outcomes

### Multiple Forces Salvo
- Extends the basic model to handle multiple combatant forces
- Complex interactions between multiple forces

### Monte Carlo Simulation
- Multi-domain combat (air, naval, submarine)
- Poisson-distributed firepower
- Binomial-distributed hit probabilities
- Adaptive reinforcement probabilities
- Morale collapse mechanics

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions and feedback, please contact:
- Email: tpires@id.uff.br or tullio.mozart@marinha.mil.br
- GitHub: [KanonStarbringer](https://github.com/KanonStarbringer) 
