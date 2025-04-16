# Salvo Equations Simulator

A comprehensive web application for naval combat modeling using various implementations of the Salvo Combat Model. This application provides interactive simulations of different naval combat scenarios, from basic salvo equations to complex Monte Carlo simulations.

## Features

- **Basic Salvo Model**: Classic implementation of the salvo equations
- **Continuous-Time Salvo**: Differential equation-based model for continuous combat simulation
- **Stochastic Salvo**: Probabilistic model with Gaussian noise in damage calculations
- **Multiple Forces Salvo**: Extension to handle multiple combatant forces
- **Monte Carlo Simulation**: Advanced model with multi-domain combat (air, naval, submarine)

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
