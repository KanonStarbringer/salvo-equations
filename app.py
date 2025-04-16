import streamlit as st
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import pandas as pd

def main():
    # Set page config
    st.set_page_config(
        page_title="Salvo Equations Simulator",
        page_icon="🚢",
        layout="wide"
    )

    # Custom CSS to style the app
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stSidebar {
            background-color: #f5f5f5;
        }
        h1 {
            color: #1E3D59;
        }
        </style>
        """, unsafe_allow_html=True)

    # Load images
    try:
        logo = Image.open("assets/CDDGN.png")
        naval_battle = Image.open("assets/naval-battle.jpg")
    except:
        st.error("Error loading images. Please make sure the images are in the assets folder.")
        logo = None
        naval_battle = None

    # Sidebar
    with st.sidebar:
        st.title("Menu")
        
        # Add CDDGN logo to sidebar
        if logo:
            st.image(logo, use_container_width=True)
        else:
            st.warning("CDDGN logo not found in assets folder")
        
        # Navigation options
        page = st.selectbox(
            "Select a Model",
            ["Home", "Salvo Equations", "Continuous-Time Salvo", "Multiple Forces Salvo", "Stochastic Salvo", "Monte Carlo", "About"]
        )

    # Main content
    if page == "Home":
        st.title("Salvo Equations Simulator")
        if naval_battle:
            st.image(naval_battle, use_container_width=True)
        st.write("""
        Welcome to the Salvo Equations Simulator app! This application allows you to explore and simulate various naval combat models.
        
        Use the sidebar to navigate between different models and learn more about the simulations.
        """)

    elif page == "Salvo Equations":
        st.title("Salvo Equations Model")
        
        # Model description
        st.write("""
        The Salvo Equations model simulates combat between two forces (Blue and Red) with the following components:
        
        - Initial forces (x₀, y₀): Fractions of Blue and Red forces (0-1)
        - Firepower (fₓ, fᵧ): Firepower per unit of Blue and Red forces
        - Interceptors (qₓ, qᵧ): Fraction of enemy fire intercepted by Blue and Red
        - Defense capacity (Cₓ, Cᵧ): Maximum fraction of enemy fire that can be absorbed before saturation
        """)
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Blue Force Parameters")
            x0 = st.slider("Initial Blue Force (x₀)", 0.0, 1.0, 1.0, 0.1)
            fx = st.slider("Blue Firepower (fₓ)", 0.0, 1.0, 0.6, 0.1)
            qx = st.slider("Blue Interceptors (qₓ)", 0.0, 1.0, 0.4, 0.1)
            cx = st.slider("Blue Defense Capacity (Cₓ)", 0.0, 1.0, 0.5, 0.1)
        
        with col2:
            st.subheader("Red Force Parameters")
            y0 = st.slider("Initial Red Force (y₀)", 0.0, 1.0, 1.0, 0.1)
            fy = st.slider("Red Firepower (fᵧ)", 0.0, 1.0, 0.5, 0.1)
            qy = st.slider("Red Interceptors (qᵧ)", 0.0, 1.0, 0.3, 0.1)
            cy = st.slider("Red Defense Capacity (Cᵧ)", 0.0, 1.0, 0.6, 0.1)
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        rounds = st.slider("Number of Rounds", 10, 100, 50, 10)
        
        # Nonlinear damage function
        def nonlinear_damage(incoming, capacity):
            if incoming <= capacity:
                return incoming
            else:
                return capacity + (incoming - capacity)**1.5
        
        # Run simulation
        if st.button("Run Simulation"):
            # Initialize forces
            x = [x0]
            y = [y0]
            
            # Simulate
            for n in range(rounds):
                # Calculate attacks
                attack_on_y = fx * x[-1] * (1 - qy)
                attack_on_x = fy * y[-1] * (1 - qx)
                
                # Apply damage
                damage_to_y = nonlinear_damage(attack_on_y, cy)
                damage_to_x = nonlinear_damage(attack_on_x, cx)
                
                # Update forces
                new_x = max(0, x[-1] - damage_to_x)
                new_y = max(0, y[-1] - damage_to_y)
                
                x.append(new_x)
                y.append(new_y)
                
                # Stop if one force is eliminated
                if new_x == 0 or new_y == 0:
                    break
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, label='Blue Force', linewidth=2)
            ax.plot(y, label='Red Force', linewidth=2)
            ax.set_title("Salvo Equations Simulation Results")
            ax.set_xlabel("Round")
            ax.set_ylabel("Fraction of Force Remaining")
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display final results
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Blue Force", f"{x[-1]:.3f}")
            with col2:
                st.metric("Final Red Force", f"{y[-1]:.3f}")

    elif page == "Multiple Forces Salvo":
        st.title("Multiple Forces Salvo Model")
        
        # Model description
        st.write("""
        The Multiple Forces Salvo model simulates combat between two forces (Blue and Red) with three types of units each:
        
        - Air forces
        - Naval forces
        - Submarine forces
        
        Each force type has different effectiveness against different target types, represented by firepower matrices.
        """)
        
        # Input parameters
        st.subheader("Blue Force Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Air Force")
            x_air = st.slider("Initial Blue Air Force", 0.0, 1.0, 1.0, 0.1)
            F_x_air_air = st.slider("Air vs Air (Fₓ₁₁)", 0.0, 1.0, 0.3, 0.1)
            F_x_air_naval = st.slider("Air vs Naval (Fₓ₁₂)", 0.0, 1.0, 0.2, 0.1)
            F_x_air_sub = st.slider("Air vs Sub (Fₓ₁₃)", 0.0, 1.0, 0.1, 0.1)
        
        with col2:
            st.write("Naval Force")
            x_naval = st.slider("Initial Blue Naval Force", 0.0, 1.0, 1.0, 0.1)
            F_x_naval_air = st.slider("Naval vs Air (Fₓ₂₁)", 0.0, 1.0, 0.4, 0.1)
            F_x_naval_naval = st.slider("Naval vs Naval (Fₓ₂₂)", 0.0, 1.0, 0.6, 0.1)
            F_x_naval_sub = st.slider("Naval vs Sub (Fₓ₂₃)", 0.0, 1.0, 0.3, 0.1)
        
        with col3:
            st.write("Submarine Force")
            x_sub = st.slider("Initial Blue Sub Force", 0.0, 1.0, 1.0, 0.1)
            F_x_sub_air = st.slider("Sub vs Air (Fₓ₃₁)", 0.0, 1.0, 0.1, 0.1)
            F_x_sub_naval = st.slider("Sub vs Naval (Fₓ₃₂)", 0.0, 1.0, 0.3, 0.1)
            F_x_sub_sub = st.slider("Sub vs Sub (Fₓ₃₃)", 0.0, 1.0, 0.5, 0.1)
        
        st.subheader("Red Force Parameters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("Air Force")
            y_air = st.slider("Initial Red Air Force", 0.0, 1.0, 1.0, 0.1)
            F_y_air_air = st.slider("Air vs Air (Fᵧ₁₁)", 0.0, 1.0, 0.2, 0.1)
            F_y_air_naval = st.slider("Air vs Naval (Fᵧ₁₂)", 0.0, 1.0, 0.4, 0.1)
            F_y_air_sub = st.slider("Air vs Sub (Fᵧ₁₃)", 0.0, 1.0, 0.2, 0.1)
        
        with col2:
            st.write("Naval Force")
            y_naval = st.slider("Initial Red Naval Force", 0.0, 1.0, 1.0, 0.1)
            F_y_naval_air = st.slider("Naval vs Air (Fᵧ₂₁)", 0.0, 1.0, 0.5, 0.1)
            F_y_naval_naval = st.slider("Naval vs Naval (Fᵧ₂₂)", 0.0, 1.0, 0.5, 0.1)
            F_y_naval_sub = st.slider("Naval vs Sub (Fᵧ₂₃)", 0.0, 1.0, 0.3, 0.1)
        
        with col3:
            st.write("Submarine Force")
            y_sub = st.slider("Initial Red Sub Force", 0.0, 1.0, 1.0, 0.1)
            F_y_sub_air = st.slider("Sub vs Air (Fᵧ₃₁)", 0.0, 1.0, 0.1, 0.1)
            F_y_sub_naval = st.slider("Sub vs Naval (Fᵧ₃₂)", 0.0, 1.0, 0.2, 0.1)
            F_y_sub_sub = st.slider("Sub vs Sub (Fᵧ₃₃)", 0.0, 1.0, 0.4, 0.1)
        
        # Defense parameters
        st.subheader("Defense Parameters")
        col1, col2 = st.columns(2)
        with col1:
            q_x = st.slider("Blue Defense Factor (qₓ)", 0.0, 1.0, 0.3, 0.1)
        with col2:
            q_y = st.slider("Red Defense Factor (qᵧ)", 0.0, 1.0, 0.25, 0.1)
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        rounds = st.slider("Number of Rounds", 10, 100, 30, 10)
        
        if st.button("Run Simulation"):
            # Initialize forces
            x = np.array([x_air, x_naval, x_sub])
            y = np.array([y_air, y_naval, y_sub])
            
            # Create firepower matrices
            F_x = np.array([
                [F_x_air_air, F_x_air_naval, F_x_air_sub],
                [F_x_naval_air, F_x_naval_naval, F_x_naval_sub],
                [F_x_sub_air, F_x_sub_naval, F_x_sub_sub]
            ])
            
            F_y = np.array([
                [F_y_air_air, F_y_air_naval, F_y_air_sub],
                [F_y_naval_air, F_y_naval_naval, F_y_naval_sub],
                [F_y_sub_air, F_y_sub_naval, F_y_sub_sub]
            ])
            
            # Initialize history
            x_history = [x.copy()]
            y_history = [y.copy()]
            
            # Run simulation
            for _ in range(rounds):
                if x.sum() == 0 or y.sum() == 0:
                    break
                
                # Calculate damage from Blue to Red
                damage_to_y = np.zeros(3)
                for i in range(3):
                    for j in range(3):
                        damage_to_y[j] += F_x[i][j] * x[i] * (1 - q_y)
                
                # Calculate damage from Red to Blue
                damage_to_x = np.zeros(3)
                for i in range(3):
                    for j in range(3):
                        damage_to_x[j] += F_y[i][j] * y[i] * (1 - q_x)
                
                # Update forces
                x = np.maximum(0, x - damage_to_x)
                y = np.maximum(0, y - damage_to_y)
                
                x_history.append(x.copy())
                y_history.append(y.copy())
            
            # Convert history to numpy arrays for plotting
            x_history = np.array(x_history)
            y_history = np.array(y_history)
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
            
            # Blue forces plot
            ax1.plot(x_history[:, 0], label='Air Force', linewidth=2)
            ax1.plot(x_history[:, 1], label='Naval Force', linewidth=2)
            ax1.plot(x_history[:, 2], label='Submarine Force', linewidth=2)
            ax1.set_title("Blue Forces Evolution")
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Fraction of Force Remaining")
            ax1.grid(True)
            ax1.legend()
            
            # Red forces plot
            ax2.plot(y_history[:, 0], label='Air Force', linewidth=2)
            ax2.plot(y_history[:, 1], label='Naval Force', linewidth=2)
            ax2.plot(y_history[:, 2], label='Submarine Force', linewidth=2)
            ax2.set_title("Red Forces Evolution")
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Fraction of Force Remaining")
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display final results
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Blue Forces")
                st.metric("Air Force", f"{x[0]:.3f}")
                st.metric("Naval Force", f"{x[1]:.3f}")
                st.metric("Submarine Force", f"{x[2]:.3f}")
            
            with col2:
                st.write("Red Forces")
                st.metric("Air Force", f"{y[0]:.3f}")
                st.metric("Naval Force", f"{y[1]:.3f}")
                st.metric("Submarine Force", f"{y[2]:.3f}")

    elif page == "About":
        st.title("About")
        st.write("""
        This application was developed to simulate and visualize various naval combat models, including the Salvo Equations model.
        
        ## References
        
        1. Hughes, W. P. (1995). Fleet Tactics and Coastal Combat. Naval Institute Press.
        2. Hughes, W. P. (2000). Fleet Tactics: Theory and Practice. Naval Institute Press.
        3. Hughes, W. P. (2002). Fleet Tactics and Naval Operations. Naval Institute Press.
        """)

    elif page == "Continuous-Time Salvo":
        st.title("Continuous-Time Salvo Model")
        
        # Model description
        st.write("""
        The Continuous-Time Salvo model simulates combat between two forces (Blue and Red) with the following components:
        
        - Initial forces (x₀, y₀): Fractions of Blue and Red forces (0-1)
        - Firepower (fₓ, fᵧ): Firepower per unit of Blue and Red forces
        - Interceptors (qₓ, qᵧ): Fraction of enemy fire intercepted by Blue and Red
        - Defense capacity (Cₓ, Cᵧ): Maximum fraction of enemy fire that can be absorbed before saturation
        """)
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Blue Force Parameters")
            x0 = st.slider("Initial Blue Force (x₀)", 0.0, 1.0, 1.0, 0.1)
            fx = st.slider("Blue Firepower (fₓ)", 0.0, 1.0, 0.6, 0.1)
            qx = st.slider("Blue Interceptors (qₓ)", 0.0, 1.0, 0.4, 0.1)
            cx = st.slider("Blue Defense Capacity (Cₓ)", 0.0, 1.0, 0.5, 0.1)
        
        with col2:
            st.subheader("Red Force Parameters")
            y0 = st.slider("Initial Red Force (y₀)", 0.0, 1.0, 1.0, 0.1)
            fy = st.slider("Red Firepower (fᵧ)", 0.0, 1.0, 0.5, 0.1)
            qy = st.slider("Red Interceptors (qᵧ)", 0.0, 1.0, 0.3, 0.1)
            cy = st.slider("Red Defense Capacity (Cᵧ)", 0.0, 1.0, 0.6, 0.1)
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        t_max = st.slider("Maximum Simulation Time", 1.0, 20.0, 10.0)
        
        def salvo_equations(x, y, fx, fy, qx, qy):
            """Calculate the next state of the salvo equations model."""
            dx = -fy * y * (1 - qx)
            dy = -fx * x * (1 - qy)
            return dx, dy

        def continuous_salvo(t, z, fx, fy, qx, qy):
            """Calculate the derivatives for the continuous-time salvo model."""
            x, y = z
            dxdt = -fy * y * (1 - qx)
            dydt = -fx * x * (1 - qy)
            return [dxdt, dydt]

        def run_salvo_simulation(fx, fy, qx, qy, steps=10):
            """Run the discrete-time salvo equations simulation."""
            x = [1.0]
            y = [1.0]
            
            for _ in range(steps):
                dx, dy = salvo_equations(x[-1], y[-1], fx, fy, qx, qy)
                x.append(max(0, x[-1] + dx))
                y.append(max(0, y[-1] + dy))
            
            return x, y

        def run_continuous_simulation(fx, fy, qx, qy, t_span=(0, 10), t_eval=None):
            """Run the continuous-time salvo model simulation."""
            if t_eval is None:
                t_eval = np.linspace(t_span[0], t_span[1], 200)
            
            z0 = [1.0, 1.0]  # Initial conditions
            sol = solve_ivp(continuous_salvo, t_span, z0, args=(fx, fy, qx, qy), t_eval=t_eval)
            return sol.t, sol.y[0], sol.y[1]

        if st.button("Run Simulation"):
            t, x, y = run_continuous_simulation(fx, fy, qx, qy, t_span=(0, t_max))
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t, x, label="Blue Force", linewidth=2)
            ax.plot(t, y, label="Red Force", linewidth=2)
            ax.set_title("Continuous-Time Salvo Simulation Results")
            ax.set_xlabel("Time")
            ax.set_ylabel("Fraction of Force Remaining")
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display final results
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Blue Force", f"{x[-1]:.3f}")
            with col2:
                st.metric("Final Red Force", f"{y[-1]:.3f}")

    elif page == "Stochastic Salvo":
        st.title("Stochastic Salvo Model")
        
        # Model description
        st.write("""
        The Stochastic Salvo model extends the basic Salvo Equations by incorporating Gaussian noise in the damage calculations.
        This model accounts for the inherent uncertainty in combat outcomes.
        
        Key components:
        - Initial forces (x₀, y₀): Fractions of Blue and Red forces (0-1)
        - Firepower (fₓ, fᵧ): Firepower per unit of Blue and Red forces
        - Interceptors (qₓ, qᵧ): Fraction of enemy fire intercepted by Blue and Red
        - Noise level (σ): Standard deviation of the Gaussian noise in damage calculations
        """)
        
        # Input parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Blue Force Parameters")
            x0 = st.slider("Initial Blue Force (x₀)", 0.0, 1.0, 1.0, 0.1)
            fx = st.slider("Blue Firepower (fₓ)", 0.0, 1.0, 0.6, 0.1)
            qx = st.slider("Blue Interceptors (qₓ)", 0.0, 1.0, 0.4, 0.1)
        
        with col2:
            st.subheader("Red Force Parameters")
            y0 = st.slider("Initial Red Force (y₀)", 0.0, 1.0, 1.0, 0.1)
            fy = st.slider("Red Firepower (fᵧ)", 0.0, 1.0, 0.5, 0.1)
            qy = st.slider("Red Interceptors (qᵧ)", 0.0, 1.0, 0.3, 0.1)
        
        # Noise parameter
        st.subheader("Stochastic Parameters")
        sigma = st.slider("Noise Level (σ)", 0.0, 0.5, 0.1, 0.05)
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        rounds = st.slider("Number of Rounds", 10, 100, 50, 10)
        num_simulations = st.slider("Number of Simulations", 1, 20, 5, 1)
        
        # Run simulation
        if st.button("Run Simulation"):
            # Initialize arrays for storing results
            all_x = np.zeros((num_simulations, rounds + 1))
            all_y = np.zeros((num_simulations, rounds + 1))
            
            # Run multiple simulations
            for sim in range(num_simulations):
                # Initialize forces
                x = [x0]
                y = [y0]
                
                # Simulate
                for n in range(rounds):
                    # Calculate attacks
                    attack_on_y = fx * x[-1] * (1 - qy)
                    attack_on_x = fy * y[-1] * (1 - qx)
                    
                    # Add Gaussian noise to damage
                    noise_y = np.random.normal(0, sigma)
                    noise_x = np.random.normal(0, sigma)
                    
                    # Apply damage with noise
                    damage_to_y = max(0, attack_on_y + noise_y)
                    damage_to_x = max(0, attack_on_x + noise_x)
                    
                    # Update forces
                    new_x = max(0, x[-1] - damage_to_x)
                    new_y = max(0, y[-1] - damage_to_y)
                    
                    x.append(new_x)
                    y.append(new_y)
                    
                    # Stop if one force is eliminated
                    if new_x == 0 or new_y == 0:
                        break
                
                # Store results
                all_x[sim, :len(x)] = x
                all_y[sim, :len(y)] = y
            
            # Plot results
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Plot individual simulations with transparency
            for sim in range(num_simulations):
                ax.plot(all_x[sim], 'b-', alpha=0.2, linewidth=1)
                ax.plot(all_y[sim], 'r-', alpha=0.2, linewidth=1)
            
            # Plot mean trajectories
            mean_x = np.mean(all_x, axis=0)
            mean_y = np.mean(all_y, axis=0)
            ax.plot(mean_x, 'b-', label='Blue Force (Mean)', linewidth=2)
            ax.plot(mean_y, 'r-', label='Red Force (Mean)', linewidth=2)
            
            ax.set_title("Stochastic Salvo Equations Simulation Results")
            ax.set_xlabel("Round")
            ax.set_ylabel("Fraction of Force Remaining")
            ax.grid(True)
            ax.legend()
            
            st.pyplot(fig)
            
            # Display final results
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Blue Force (Mean)", f"{mean_x[-1]:.3f}")
                st.metric("Blue Force Std Dev", f"{np.std(all_x[:, -1]):.3f}")
            with col2:
                st.metric("Final Red Force (Mean)", f"{mean_y[-1]:.3f}")
                st.metric("Red Force Std Dev", f"{np.std(all_y[:, -1]):.3f}")

    elif page == "Monte Carlo":
        st.title("Monte Carlo Naval Combat Model")
        
        # Model description
        st.write("""
        The Monte Carlo Naval Combat Model simulates multi-domain combat (air, naval, submarine) with stochastic elements:
        
        Key components:
        - Multiple domains (air, naval, submarine)
        - Poisson-distributed firepower generation
        - Binomial-distributed hit probabilities
        - Adaptive reinforcement probabilities
        - Morale collapse mechanics
        
        The model uses the following equations:
        
        1. Firepower Generation:
        $S_{i}^{(d)} \\sim \\text{Poisson}(\\lambda_{i}^{(d)})$
        
        2. Hit Probability:
        $H_{i \\rightarrow j}^{(d_k \\rightarrow d_l)} \\sim \\text{Binomial}\\left(S_{i}^{(d_k)},\\; p_{\\text{hit},\\,i}^{(d_k \\rightarrow d_l)} \\cdot (1 - p_{\\text{int},\\,j})\\right)$
        
        3. Force Update:
        $U_j^{(d_l)}(t+1) = \\max\\left(0,\\; U_j^{(d_l)}(t) - \\sum_{d_k} H_{i \\rightarrow j}^{(d_k \\rightarrow d_l)} \\right)$
        
        4. Reinforcement Probability:
        $P_{\\text{reforço},\\,i}^{(d)} = \\min\\left(0.02 + 0.1 \\cdot \\left(1 - \\frac{\\sum_d U_i^{(d)}(t)}{\\sum_d U_i^{(d)}(0)}\\right),\\; 1.0\\right)$
        """)
        
        # Simulation parameters
        st.subheader("Simulation Parameters")
        col1, col2 = st.columns(2)
        
        with col1:
            num_simulations = st.slider("Number of Simulations", 10, 200, 100, 10)
            max_rounds = st.slider("Maximum Rounds", 10, 100, 50, 10)
            morale_threshold = st.slider("Morale Threshold", 0.0, 0.5, 0.2, 0.05)
            morale_collapse_chance = st.slider("Morale Collapse Chance", 0.0, 0.5, 0.2, 0.05)
        
        with col2:
            avg_shots_blue = st.slider("Blue Average Shots", 0.5, 2.0, 1.2, 0.1)
            avg_shots_red = st.slider("Red Average Shots", 0.5, 2.0, 1.0, 0.1)
            p_int_blue = st.slider("Blue Interception Probability", 0.0, 0.5, 0.3, 0.05)
            p_int_red = st.slider("Red Interception Probability", 0.0, 0.5, 0.3, 0.05)
        
        # Initial forces
        st.subheader("Initial Forces")
        domains = ["Air", "Naval", "Submarine"]
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("Blue Force")
            blue_initial = []
            for domain in domains:
                blue_initial.append(st.number_input(f"Blue {domain} Forces", 0, 100, 20, 1))
        
        with col2:
            st.write("Red Force")
            red_initial = []
            for domain in domains:
                red_initial.append(st.number_input(f"Red {domain} Forces", 0, 100, 20, 1))
        
        # Hit probabilities matrix
        st.subheader("Hit Probabilities Matrix")
        st.write("Probability of hits between domains (rows: attacker, columns: defender)")
        
        # Blue hit probabilities
        st.write("Blue Force Hit Probabilities")
        p_hit_blue = np.zeros((3, 3))
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                p_hit_blue[i,j] = cols[j].slider(
                    f"Blue {domains[i]} → {domains[j]}", 
                    0.0, 1.0, 
                    [0.6, 0.4, 0.2, 0.5, 0.6, 0.3, 0.2, 0.3, 0.7][i*3 + j], 
                    0.1
                )
        
        # Red hit probabilities
        st.write("Red Force Hit Probabilities")
        p_hit_red = np.zeros((3, 3))
        for i in range(3):
            cols = st.columns(3)
            for j in range(3):
                p_hit_red[i,j] = cols[j].slider(
                    f"Red {domains[i]} → {domains[j]}", 
                    0.0, 1.0, 
                    [0.6, 0.4, 0.2, 0.5, 0.6, 0.3, 0.2, 0.3, 0.7][i*3 + j], 
                    0.1
                )
        
        # Run simulation
        if st.button("Run Simulation"):
            # Initialize arrays for storing results
            all_blue = np.zeros((num_simulations, max_rounds + 1, 3))
            all_red = np.zeros((num_simulations, max_rounds + 1, 3))
            
            # Run multiple simulations
            for sim in range(num_simulations):
                # Initialize forces
                blue = np.array(blue_initial)
                red = np.array(red_initial)
                all_blue[sim, 0] = blue
                all_red[sim, 0] = red
                
                # Simulate
                for round in range(max_rounds):
                    # Check for morale collapse
                    blue_ratio = np.sum(blue) / np.sum(blue_initial)
                    red_ratio = np.sum(red) / np.sum(red_initial)
                    
                    if blue_ratio < morale_threshold and np.random.random() < morale_collapse_chance:
                        blue = np.zeros_like(blue)
                    if red_ratio < morale_threshold and np.random.random() < morale_collapse_chance:
                        red = np.zeros_like(red)
                    
                    # Calculate shots (Poisson distribution)
                    blue_shots = np.random.poisson(avg_shots_blue * blue)
                    red_shots = np.random.poisson(avg_shots_red * red)
                    
                    # Calculate hits (Binomial distribution)
                    blue_hits = np.zeros_like(blue)
                    red_hits = np.zeros_like(red)
                    
                    for i in range(3):  # Attacker domain
                        for j in range(3):  # Defender domain
                            if blue[i] > 0:
                                p_hit = p_hit_blue[i,j] * (1 - p_int_red)
                                blue_hits[j] += np.random.binomial(blue_shots[i], p_hit)
                            if red[i] > 0:
                                p_hit = p_hit_red[i,j] * (1 - p_int_blue)
                                red_hits[j] += np.random.binomial(red_shots[i], p_hit)
                    
                    # Update forces
                    blue = np.maximum(0, blue - red_hits)
                    red = np.maximum(0, red - blue_hits)
                    
                    # Reinforcement
                    for i in range(3):
                        if blue[i] > 0:
                            p_reinforce = min(0.02 + 0.1 * (1 - np.sum(blue)/np.sum(blue_initial)), 1.0)
                            if np.random.random() < p_reinforce:
                                blue[i] += 1
                        if red[i] > 0:
                            p_reinforce = min(0.02 + 0.1 * (1 - np.sum(red)/np.sum(red_initial)), 1.0)
                            if np.random.random() < p_reinforce:
                                red[i] += 1
                    
                    # Store results
                    all_blue[sim, round + 1] = blue
                    all_red[sim, round + 1] = red
                    
                    # Stop if one side is eliminated
                    if np.sum(blue) == 0 or np.sum(red) == 0:
                        break
            
            # Plot results
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Plot mean trajectories for each domain
            for i, domain in enumerate(domains):
                mean_blue = np.mean(all_blue[:, :, i], axis=0)
                mean_red = np.mean(all_red[:, :, i], axis=0)
                
                ax1.plot(mean_blue, 'b-', label=f'Blue {domain} (Mean)', alpha=0.7)
                ax1.plot(mean_red, 'r-', label=f'Red {domain} (Mean)', alpha=0.7)
            
            ax1.set_title("Mean Force Levels by Domain")
            ax1.set_xlabel("Round")
            ax1.set_ylabel("Number of Units")
            ax1.grid(True)
            ax1.legend()
            
            # Plot total forces
            total_blue = np.sum(all_blue, axis=2)
            total_red = np.sum(all_red, axis=2)
            
            mean_total_blue = np.mean(total_blue, axis=0)
            mean_total_red = np.mean(total_red, axis=0)
            
            ax2.plot(mean_total_blue, 'b-', label='Blue Total (Mean)', linewidth=2)
            ax2.plot(mean_total_red, 'r-', label='Red Total (Mean)', linewidth=2)
            
            # Plot individual simulations with transparency
            for sim in range(num_simulations):
                ax2.plot(total_blue[sim], 'b-', alpha=0.1, linewidth=1)
                ax2.plot(total_red[sim], 'r-', alpha=0.1, linewidth=1)
            
            ax2.set_title("Total Force Levels")
            ax2.set_xlabel("Round")
            ax2.set_ylabel("Total Units")
            ax2.grid(True)
            ax2.legend()
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Display final results
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("Blue Force")
                for i, domain in enumerate(domains):
                    mean = np.mean(all_blue[:, -1, i])
                    std = np.std(all_blue[:, -1, i])
                    st.metric(f"{domain} Forces", f"{mean:.1f} ± {std:.1f}")
            
            with col2:
                st.write("Red Force")
                for i, domain in enumerate(domains):
                    mean = np.mean(all_red[:, -1, i])
                    std = np.std(all_red[:, -1, i])
                    st.metric(f"{domain} Forces", f"{mean:.1f} ± {std:.1f}")

if __name__ == "__main__":
    main() 