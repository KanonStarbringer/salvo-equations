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
            ["Home", "Salvo Equations", "Classical Model - Nonlinear damage", "Continuous-Time Salvo", "Multiple Forces Salvo", "Stochastic Salvo", "Monte Carlo", "About"]
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

    elif page == "Classical Model - Nonlinear damage":
        st.title("Classical Model - Nonlinear Damage")
        st.write("""
        This model simulates combat between two forces (Blue and Red) using nonlinear (saturation) damage.
        """)
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
        st.subheader("Simulation Parameters")
        rounds = st.slider("Number of Rounds", 10, 100, 50, 10)
        fire_mode = st.radio("Fire Order", ["Simultaneous", "Blue fires first", "Red fires first"])
        def nonlinear_damage(incoming, capacity):
            if incoming <= capacity:
                return incoming
            else:
                return capacity + (incoming - capacity)**1.5
        if st.button("Run Simulation", key="run_nonlinear"):
            x = [x0]
            y = [y0]
            for n in range(rounds):
                if fire_mode == "Simultaneous":
                    attack_on_y = fx * x[-1] * (1 - qy)
                    attack_on_x = fy * y[-1] * (1 - qx)
                    damage_to_y = nonlinear_damage(attack_on_y, cy)
                    damage_to_x = nonlinear_damage(attack_on_x, cx)
                    new_x = max(0, x[-1] - damage_to_x)
                    new_y = max(0, y[-1] - damage_to_y)
                elif fire_mode == "Blue fires first":
                    # Blue attacks Red
                    attack_on_y = fx * x[-1] * (1 - qy)
                    damage_to_y = nonlinear_damage(attack_on_y, cy)
                    temp_y = max(0, y[-1] - damage_to_y)
                    # Red attacks Blue with reduced force
                    attack_on_x = fy * temp_y * (1 - qx)
                    damage_to_x = nonlinear_damage(attack_on_x, cx)
                    new_x = max(0, x[-1] - damage_to_x)
                    new_y = temp_y
                else:  # Red fires first
                    attack_on_x = fy * y[-1] * (1 - qx)
                    damage_to_x = nonlinear_damage(attack_on_x, cx)
                    temp_x = max(0, x[-1] - damage_to_x)
                    attack_on_y = fx * temp_x * (1 - qy)
                    damage_to_y = nonlinear_damage(attack_on_y, cy)
                    new_y = max(0, y[-1] - damage_to_y)
                    new_x = temp_x
                x.append(new_x)
                y.append(new_y)
                if new_x == 0 or new_y == 0:
                    break
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(x, label='Blue Force', linewidth=2)
            ax.plot(y, label='Red Force', linewidth=2)
            ax.set_title("Results - Classical Model with Nonlinear Damage")
            ax.set_xlabel("Round")
            ax.set_ylabel("Fraction of Force Remaining")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.subheader("Final Results")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Final Blue Force", f"{x[-1]:.3f}")
            with col2:
                st.metric("Final Red Force", f"{y[-1]:.3f}")

    elif page == "Salvo Equations":
        st.title("Salvo Equations Model (Hughes)")
        st.write("""
        This model implements the Salvo Equations as described by Hughes:
        """)
        st.latex(r"\Delta B = \frac{\alpha A - b_3 B}{b_1}, \quad \Delta A = \frac{\beta B - a_3 A}{a_1}")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Force A (blue) initial parameters")
            A0 = st.number_input("A initial units", 0.0, 1000.0, 20.0, 1.0)
            alpha = st.number_input("A well aimed missiles (α)", 0.0, 1000.0, 10.0, 1.0)
            a1 = st.number_input("Necessary hits to destroy 1 A (a₁)", 0.1, 1000.0, 2.0, 0.1)
            a3 = st.number_input("Missiles destroyed by each A (a₃)", 0.0, 1000.0, 1.0, 1.0)
        with col2:
            st.subheader("Force B (red) initial parameters")
            B0 = st.number_input("B initial units", 0.0, 1000.0, 20.0, 1.0)
            beta = st.number_input("B well aimed missiles (β)", 0.0, 1000.0, 10.0, 1.0)
            b1 = st.number_input("Necessary hits to destroy 1 B (b₁)", 0.1, 1000.0, 2.0, 0.1)
            b3 = st.number_input("Missiles destroyed by each B (b₃)", 0.0, 1000.0, 1.0, 1.0)
        st.subheader("Simulation Parameters")
        rounds = st.slider("Number or rounds", 1, 100, 20, 1)
        fire_mode = st.radio("Fire Order", ["Simultaneous", "Blue fires first", "Red fires first"])
        if st.button("Run Sim", key="run_hughes"):
            A = [A0]
            B = [B0]
            for n in range(rounds):
                if fire_mode == "Simultaneous":
                    delta_B = max(0, (alpha * A[-1] - b3 * B[-1]) / b1)
                    delta_A = max(0, (beta * B[-1] - a3 * A[-1]) / a1)
                    new_A = max(0, A[-1] - delta_A)
                    new_B = max(0, B[-1] - delta_B)
                elif fire_mode == "Blue fires first":
                    delta_B = max(0, (alpha * A[-1] - b3 * B[-1]) / b1)
                    temp_B = max(0, B[-1] - delta_B)
                    delta_A = max(0, (beta * temp_B - a3 * A[-1]) / a1)
                    new_A = max(0, A[-1] - delta_A)
                    new_B = temp_B
                else:  # Red fires first
                    delta_A = max(0, (beta * B[-1] - a3 * A[-1]) / a1)
                    temp_A = max(0, A[-1] - delta_A)
                    delta_B = max(0, (alpha * temp_A - b3 * B[-1]) / b1)
                    new_B = max(0, B[-1] - delta_B)
                    new_A = temp_A
                A.append(new_A)
                B.append(new_B)
                if new_A <= 0 or new_B <= 0:
                    break
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(A, label='Force A (blue)', linewidth=2)
            ax.plot(B, label='Force B (red)', linewidth=2)
            ax.set_title("Results - Salvo Equations (Hughes)")
            ax.set_xlabel("Round")
            ax.set_ylabel("Remaining Units")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
            st.subheader("Final Score")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Force A remaining (blue)", f"{A[-1]:.2f}")
            with col2:
                st.metric("Force B  remaining (red)", f"{B[-1]:.2f}")

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
        fire_mode = st.radio("Fire Order", ["Simultaneous", "Blue fires first", "Red fires first"])
        
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
                
                if fire_mode == "Simultaneous":
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
                    
                elif fire_mode == "Blue fires first":
                    # Blue attacks Red first
                    damage_to_y = np.zeros(3)
                    for i in range(3):
                        for j in range(3):
                            damage_to_y[j] += F_x[i][j] * x[i] * (1 - q_y)
                    
                    temp_y = np.maximum(0, y - damage_to_y)
                    
                    # Red attacks Blue with reduced force
                    damage_to_x = np.zeros(3)
                    for i in range(3):
                        for j in range(3):
                            damage_to_x[j] += F_y[i][j] * temp_y[i] * (1 - q_x)
                    
                    x = np.maximum(0, x - damage_to_x)
                    y = temp_y
                    
                else:  # Red fires first
                    # Red attacks Blue first
                    damage_to_x = np.zeros(3)
                    for i in range(3):
                        for j in range(3):
                            damage_to_x[j] += F_y[i][j] * y[i] * (1 - q_x)
                    
                    temp_x = np.maximum(0, x - damage_to_x)
                    
                    # Blue attacks Red with reduced force
                    damage_to_y = np.zeros(3)
                    for i in range(3):
                        for j in range(3):
                            damage_to_y[j] += F_x[i][j] * temp_x[i] * (1 - q_y)
                    
                    x = temp_x
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
        fire_mode = st.radio("Fire Order", ["Simultaneous", "Blue fires first", "Red fires first"])

        def continuous_salvo(t, z, fx, fy, qx, qy, fire_mode="Simultaneous"):
            x, y = z
            # Garante que as forças não fiquem negativas
            x = max(0, x)
            y = max(0, y)
            
            if fire_mode == "Simultaneous":
                dxdt = -fy * y * (1 - qx) if x > 0 else 0
                dydt = -fx * x * (1 - qy) if y > 0 else 0
            elif fire_mode == "Blue fires first":
                dxdt = -fy * y * (1 - qx) if x > 0 else 0
                dydt = -fx * x * (1 - qy) if y > 0 else 0
                # Ajusta o dano para considerar o ataque do azul primeiro
                if x > 0:
                    temp_y = max(0, y - fx * x * (1 - qy) * t)
                    dydt = -fx * x * (1 - qy) if temp_y > 0 else 0
            else:  # Red fires first
                dxdt = -fy * y * (1 - qx) if x > 0 else 0
                dydt = -fx * x * (1 - qy) if y > 0 else 0
                # Ajusta o dano para considerar o ataque do vermelho primeiro
                if y > 0:
                    temp_x = max(0, x - fy * y * (1 - qx) * t)
                    dxdt = -fy * y * (1 - qx) if temp_x > 0 else 0
            
            return [dxdt, dydt]

        def run_continuous_simulation(x0, y0, fx, fy, qx, qy, t_span=(0, 10), t_eval=None, fire_mode="Simultaneous"):
            if t_eval is None:
                t_eval = np.linspace(t_span[0], t_span[1], 200)
            z0 = [x0, y0]  # Usa os valores iniciais dos sliders
            sol = solve_ivp(continuous_salvo, t_span, z0, args=(fx, fy, qx, qy, fire_mode), t_eval=t_eval)
            # Garante que as forças não fiquem negativas
            x = np.maximum(sol.y[0], 0)
            y = np.maximum(sol.y[1], 0)
            return sol.t, x, y

        if st.button("Run Simulation"):
            t, x, y = run_continuous_simulation(x0, y0, fx, fy, qx, qy, t_span=(0, t_max), fire_mode=fire_mode)
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(t, x, label="Blue Force", linewidth=2)
            ax.plot(t, y, label="Red Force", linewidth=2)
            ax.set_title("Continuous-Time Salvo Simulation Results")
            ax.set_xlabel("Time")
            ax.set_ylabel("Fraction of Force Remaining")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
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
        num_simulations = st.slider("Number of Simulations", 10, 1000, 100, 10)
        fire_mode = st.radio("Fire Order", ["Simultaneous", "Blue fires first", "Red fires first"])
        
        # Run simulation
        if st.button("Run Simulation"):
            # Initialize arrays for storing results
            all_x = np.zeros((num_simulations, rounds + 1))
            all_y = np.zeros((num_simulations, rounds + 1))
            final_rounds = np.zeros(num_simulations)
            blue_wins = 0
            red_wins = 0
            draw = 0
            
            # Track domain effectiveness
            domain_effectiveness = {
                'blue': {'air': 0, 'naval': 0, 'submarine': 0},
                'red': {'air': 0, 'naval': 0, 'submarine': 0}
            }
            
            # Run multiple simulations
            for sim in range(num_simulations):
                # Initialize forces
                x = [x0]
                y = [y0]
                
                # Simulate
                for n in range(rounds):
                    if fire_mode == "Simultaneous":
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
                        
                    elif fire_mode == "Blue fires first":
                        # Blue attacks Red first
                        attack_on_y = fx * x[-1] * (1 - qy)
                        noise_y = np.random.normal(0, sigma)
                        damage_to_y = max(0, attack_on_y + noise_y)
                        temp_y = max(0, y[-1] - damage_to_y)
                        
                        # Red attacks Blue with reduced force
                        attack_on_x = fy * temp_y * (1 - qx)
                        noise_x = np.random.normal(0, sigma)
                        damage_to_x = max(0, attack_on_x + noise_x)
                        new_x = max(0, x[-1] - damage_to_x)
                        new_y = temp_y
                        
                    else:  # Red fires first
                        # Red attacks Blue first
                        attack_on_x = fy * y[-1] * (1 - qx)
                        noise_x = np.random.normal(0, sigma)
                        damage_to_x = max(0, attack_on_x + noise_x)
                        temp_x = max(0, x[-1] - damage_to_x)
                        
                        # Blue attacks Red with reduced force
                        attack_on_y = fx * temp_x * (1 - qy)
                        noise_y = np.random.normal(0, sigma)
                        damage_to_y = max(0, attack_on_y + noise_y)
                        new_y = max(0, y[-1] - damage_to_y)
                        new_x = temp_x
                    
                    x.append(new_x)
                    y.append(new_y)
                    
                    # Stop if one force is eliminated
                    if new_x == 0 or new_y == 0:
                        break
                
                # Store results
                all_x[sim, :len(x)] = x
                all_y[sim, :len(y)] = y
                final_rounds[sim] = len(x) - 1
                
                # Determine winner
                if new_x > 0 and new_y == 0:
                    blue_wins += 1
                elif new_y > 0 and new_x == 0:
                    red_wins += 1
                else:
                    draw += 1
            
            # Calculate victory probabilities
            blue_win_prob = blue_wins / num_simulations
            red_win_prob = red_wins / num_simulations
            draw_prob = draw / num_simulations
            
            # Calculate mean survival time
            mean_survival_time = np.mean(final_rounds)
            std_survival_time = np.std(final_rounds)
            
            # Calculate force ratio over time
            force_ratio = np.zeros(rounds + 1)
            force_ratio_std = np.zeros(rounds + 1)
            
            for t in range(rounds + 1):
                valid_sims = 0
                ratios = []
                
                for sim in range(num_simulations):
                    if t < len(all_x[sim]) and t < len(all_y[sim]):
                        if all_y[sim, t] > 0:  # Avoid division by zero
                            ratios.append(all_x[sim, t] / all_y[sim, t])
                            valid_sims += 1
                
                if valid_sims > 0:
                    force_ratio[t] = np.mean(ratios)
                    force_ratio_std[t] = np.std(ratios)
            
            # Plot results
            st.subheader("Simulation Results")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4 = st.tabs(["Force Evolution", "Victory Analysis", "Survival Time", "Force Ratio"])
            
            with tab1:
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                
                # Plot individual simulations with transparency
                for sim in range(num_simulations):
                    ax1.plot(all_x[sim], 'b-', alpha=0.1, linewidth=1)
                    ax1.plot(all_y[sim], 'r-', alpha=0.1, linewidth=1)
                
                # Plot mean trajectories
                mean_x = np.mean(all_x, axis=0)
                mean_y = np.mean(all_y, axis=0)
                ax1.plot(mean_x, 'b-', label='Blue Force (Mean)', linewidth=2)
                ax1.plot(mean_y, 'r-', label='Red Force (Mean)', linewidth=2)
                
                # Add confidence intervals
                std_x = np.std(all_x, axis=0)
                std_y = np.std(all_y, axis=0)
                
                ax1.fill_between(range(len(mean_x)), 
                                mean_x - std_x, 
                                mean_x + std_x, 
                                color='blue', alpha=0.2)
                ax1.fill_between(range(len(mean_y)), 
                                mean_y - std_y, 
                                mean_y + std_y, 
                                color='red', alpha=0.2)
                
                ax1.set_title("Force Evolution Over Time")
                ax1.set_xlabel("Round")
                ax1.set_ylabel("Fraction of Force Remaining")
                ax1.grid(True)
                ax1.legend()
                
                st.pyplot(fig1)
            
            with tab2:
                # Victory probability pie chart
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                labels = ['Blue Victory', 'Red Victory', 'Draw']
                sizes = [blue_win_prob, red_win_prob, draw_prob]
                colors = ['blue', 'red', 'gray']
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')
                ax2.set_title("Victory Probability")
                
                st.pyplot(fig2)
                
                # Display victory probabilities
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Blue Victory Probability", f"{blue_win_prob:.1%}")
                with col2:
                    st.metric("Red Victory Probability", f"{red_win_prob:.1%}")
                with col3:
                    st.metric("Draw Probability", f"{draw_prob:.1%}")
            
            with tab3:
                # Survival time histogram
                fig3, ax3 = plt.subplots(figsize=(10, 6))
                ax3.hist(final_rounds, bins=20, alpha=0.7, color='green')
                ax3.axvline(mean_survival_time, color='red', linestyle='dashed', linewidth=2, 
                           label=f'Mean: {mean_survival_time:.1f} rounds')
                ax3.set_title("Distribution of Combat Duration")
                ax3.set_xlabel("Number of Rounds")
                ax3.set_ylabel("Frequency")
                ax3.grid(True)
                ax3.legend()
                
                st.pyplot(fig3)
                
                # Display survival time statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Survival Time", f"{mean_survival_time:.1f} rounds")
                with col2:
                    st.metric("Std Dev of Survival Time", f"{std_survival_time:.1f} rounds")
            
            with tab4:
                # Force ratio plot
                fig4, ax4 = plt.subplots(figsize=(10, 6))
                ax4.plot(force_ratio, 'g-', label='Blue/Red Force Ratio', linewidth=2)
                ax4.fill_between(range(len(force_ratio)), 
                                force_ratio - force_ratio_std, 
                                force_ratio + force_ratio_std, 
                                color='green', alpha=0.2)
                ax4.axhline(y=1.0, color='black', linestyle='--', label='Equal Forces')
                ax4.set_title("Force Ratio Evolution (Blue/Red)")
                ax4.set_xlabel("Round")
                ax4.set_ylabel("Force Ratio")
                ax4.grid(True)
                ax4.legend()
                
                st.pyplot(fig4)
                
                # Display force ratio statistics
                st.write("Force ratio > 1 indicates Blue advantage, < 1 indicates Red advantage")
                st.metric("Final Force Ratio", f"{force_ratio[min(len(force_ratio)-1, int(mean_survival_time))]:.2f}")
            
            # Sensitivity analysis
            st.subheader("Parameter Sensitivity Analysis")
            
            # Create a simple sensitivity analysis by running simulations with slightly different parameters
            sensitivity_params = [
                ("Blue Firepower", fx, 0.1),
                ("Red Firepower", fy, 0.1),
                ("Blue Interceptors", qx, 0.1),
                ("Red Interceptors", qy, 0.1),
                ("Noise Level", sigma, 0.05)
            ]
            
            sensitivity_results = []
            
            for param_name, base_value, delta in sensitivity_params:
                # Run with increased parameter
                increased_value = base_value + delta
                increased_wins = 0
                
                for sim in range(100):  # Use fewer simulations for sensitivity analysis
                    x = [x0]
                    y = [y0]
                    
                    for n in range(rounds):
                        # Adjust the parameter being tested
                        if param_name == "Blue Firepower":
                            attack_on_y = increased_value * x[-1] * (1 - qy)
                            attack_on_x = fy * y[-1] * (1 - qx)
                        elif param_name == "Red Firepower":
                            attack_on_y = fx * x[-1] * (1 - qy)
                            attack_on_x = increased_value * y[-1] * (1 - qx)
                        elif param_name == "Blue Interceptors":
                            attack_on_y = fx * x[-1] * (1 - qy)
                            attack_on_x = fy * y[-1] * (1 - increased_value)
                        elif param_name == "Red Interceptors":
                            attack_on_y = fx * x[-1] * (1 - increased_value)
                            attack_on_x = fy * y[-1] * (1 - qx)
                        else:  # Noise Level
                            attack_on_y = fx * x[-1] * (1 - qy)
                            attack_on_x = fy * y[-1] * (1 - qx)
                            noise_y = np.random.normal(0, increased_value)
                            noise_x = np.random.normal(0, increased_value)
                            damage_to_y = max(0, attack_on_y + noise_y)
                            damage_to_x = max(0, attack_on_x + noise_x)
                            new_x = max(0, x[-1] - damage_to_x)
                            new_y = max(0, y[-1] - damage_to_y)
                            x.append(new_x)
                            y.append(new_y)
                            continue
                        
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
                    
                    # Determine winner
                    if new_x > 0 and new_y == 0:
                        increased_wins += 1
                
                # Run with decreased parameter
                decreased_value = max(0, base_value - delta)
                decreased_wins = 0
                
                for sim in range(100):  # Use fewer simulations for sensitivity analysis
                    x = [x0]
                    y = [y0]
                    
                    for n in range(rounds):
                        # Adjust the parameter being tested
                        if param_name == "Blue Firepower":
                            attack_on_y = decreased_value * x[-1] * (1 - qy)
                            attack_on_x = fy * y[-1] * (1 - qx)
                        elif param_name == "Red Firepower":
                            attack_on_y = fx * x[-1] * (1 - qy)
                            attack_on_x = decreased_value * y[-1] * (1 - qx)
                        elif param_name == "Blue Interceptors":
                            attack_on_y = fx * x[-1] * (1 - qy)
                            attack_on_x = fy * y[-1] * (1 - decreased_value)
                        elif param_name == "Red Interceptors":
                            attack_on_y = fx * x[-1] * (1 - decreased_value)
                            attack_on_x = fy * y[-1] * (1 - qx)
                        else:  # Noise Level
                            attack_on_y = fx * x[-1] * (1 - qy)
                            attack_on_x = fy * y[-1] * (1 - qx)
                            noise_y = np.random.normal(0, decreased_value)
                            noise_x = np.random.normal(0, decreased_value)
                            damage_to_y = max(0, attack_on_y + noise_y)
                            damage_to_x = max(0, attack_on_x + noise_x)
                            new_x = max(0, x[-1] - damage_to_x)
                            new_y = max(0, y[-1] - damage_to_y)
                            x.append(new_x)
                            y.append(new_y)
                            continue
                        
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
                    
                    # Determine winner
                    if new_x > 0 and new_y == 0:
                        decreased_wins += 1
                
                # Calculate sensitivity
                base_win_prob = blue_win_prob
                increased_win_prob = increased_wins / 100
                decreased_win_prob = decreased_wins / 100
                
                sensitivity = (increased_win_prob - decreased_win_prob) / (2 * delta)
                sensitivity_results.append((param_name, sensitivity))
            
            # Plot sensitivity results
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            param_names = [p[0] for p in sensitivity_results]
            sensitivities = [p[1] for p in sensitivity_results]
            
            # Sort by absolute sensitivity
            sorted_indices = np.argsort(np.abs(sensitivities))
            param_names = [param_names[i] for i in sorted_indices]
            sensitivities = [sensitivities[i] for i in sorted_indices]
            
            colors = ['blue' if s > 0 else 'red' for s in sensitivities]
            ax5.barh(param_names, sensitivities, color=colors)
            ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax5.set_title("Parameter Sensitivity on Blue Victory Probability")
            ax5.set_xlabel("Sensitivity (Change in Blue Victory Probability per Unit Change in Parameter)")
            ax5.grid(True, axis='x')
            
            st.pyplot(fig5)
            
            # Display sensitivity results
            st.write("Positive values indicate parameters that increase Blue's chance of victory when increased.")
            st.write("Negative values indicate parameters that decrease Blue's chance of victory when increased.")

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
        
        fire_mode = st.radio("Fire Order", ["Simultaneous", "Blue fires first", "Red fires first"])

        # Run simulation
        if st.button("Run Simulation"):
            # Initialize arrays for storing results
            all_blue = np.zeros((num_simulations, max_rounds + 1, 3))
            all_red = np.zeros((num_simulations, max_rounds + 1, 3))
            final_rounds = np.zeros(num_simulations)
            blue_wins = 0
            red_wins = 0
            draw = 0
            
            # Track domain effectiveness
            domain_effectiveness = {
                'blue': {'air': 0, 'naval': 0, 'submarine': 0},
                'red': {'air': 0, 'naval': 0, 'submarine': 0}
            }
            
            # Run multiple simulations
            for sim in range(num_simulations):
                # Initialize forces
                blue = np.array(blue_initial)
                red = np.array(red_initial)
                all_blue[sim, 0] = blue
                all_red[sim, 0] = red
                
                # Track hits by domain
                blue_hits_by_domain = np.zeros((3, 3))  # [attacker_domain, defender_domain]
                red_hits_by_domain = np.zeros((3, 3))
                
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
                    
                    if fire_mode == "Simultaneous":
                        # Calculate hits (Binomial distribution)
                        blue_hits = np.zeros_like(blue)
                        red_hits = np.zeros_like(red)
                        
                        for i in range(3):  # Attacker domain
                            for j in range(3):  # Defender domain
                                if blue[i] > 0:
                                    p_hit = p_hit_blue[i,j] * (1 - p_int_red)
                                    hits = np.random.binomial(blue_shots[i], p_hit)
                                    blue_hits[j] += hits
                                    blue_hits_by_domain[i,j] += hits
                                if red[i] > 0:
                                    p_hit = p_hit_red[i,j] * (1 - p_int_blue)
                                    hits = np.random.binomial(red_shots[i], p_hit)
                                    red_hits[j] += hits
                                    red_hits_by_domain[i,j] += hits
                        
                        # Update forces
                        blue = np.maximum(0, blue - red_hits)
                        red = np.maximum(0, red - blue_hits)
                        
                    elif fire_mode == "Blue fires first":
                        # Blue attacks Red first
                        blue_hits = np.zeros_like(blue)
                        for i in range(3):
                            for j in range(3):
                                if blue[i] > 0:
                                    p_hit = p_hit_blue[i,j] * (1 - p_int_red)
                                    hits = np.random.binomial(blue_shots[i], p_hit)
                                    blue_hits[j] += hits
                                    blue_hits_by_domain[i,j] += hits
                        
                        temp_red = np.maximum(0, red - blue_hits)
                        
                        # Red attacks Blue with reduced force
                        red_hits = np.zeros_like(red)
                        for i in range(3):
                            for j in range(3):
                                if temp_red[i] > 0:
                                    p_hit = p_hit_red[i,j] * (1 - p_int_blue)
                                    hits = np.random.binomial(red_shots[i], p_hit)
                                    red_hits[j] += hits
                                    red_hits_by_domain[i,j] += hits
                        
                        blue = np.maximum(0, blue - red_hits)
                        red = temp_red
                        
                    else:  # Red fires first
                        # Red attacks Blue first
                        red_hits = np.zeros_like(red)
                        for i in range(3):
                            for j in range(3):
                                if red[i] > 0:
                                    p_hit = p_hit_red[i,j] * (1 - p_int_blue)
                                    hits = np.random.binomial(red_shots[i], p_hit)
                                    red_hits[j] += hits
                                    red_hits_by_domain[i,j] += hits
                        
                        temp_blue = np.maximum(0, blue - red_hits)
                        
                        # Blue attacks Red with reduced force
                        blue_hits = np.zeros_like(blue)
                        for i in range(3):
                            for j in range(3):
                                if temp_blue[i] > 0:
                                    p_hit = p_hit_blue[i,j] * (1 - p_int_red)
                                    hits = np.random.binomial(blue_shots[i], p_hit)
                                    blue_hits[j] += hits
                                    blue_hits_by_domain[i,j] += hits
                        
                        blue = temp_blue
                        red = np.maximum(0, red - blue_hits)
                    
                    # Store results
                    all_blue[sim, round + 1] = blue
                    all_red[sim, round + 1] = red
                    
                    # Stop if one side is eliminated
                    if np.sum(blue) == 0 or np.sum(red) == 0:
                        break
                
                # Store final round
                final_rounds[sim] = round + 1
                
                # Determine winner
                if np.sum(blue) > 0 and np.sum(red) == 0:
                    blue_wins += 1
                elif np.sum(red) > 0 and np.sum(blue) == 0:
                    red_wins += 1
                else:
                    draw += 1
                
                # Update domain effectiveness
                for i in range(3):
                    for j in range(3):
                        domain_effectiveness['blue'][domains[i].lower()] += blue_hits_by_domain[i,j]
                        domain_effectiveness['red'][domains[i].lower()] += red_hits_by_domain[i,j]
            
            # Calculate victory probabilities
            blue_win_prob = blue_wins / num_simulations
            red_win_prob = red_wins / num_simulations
            draw_prob = draw / num_simulations
            
            # Calculate mean survival time
            mean_survival_time = np.mean(final_rounds)
            std_survival_time = np.std(final_rounds)
            
            # Normalize domain effectiveness
            for side in ['blue', 'red']:
                total_hits = sum(domain_effectiveness[side].values())
                if total_hits > 0:
                    for domain in domains:
                        domain_effectiveness[side][domain.lower()] /= total_hits
            
            # Plot results
            st.subheader("Simulation Results")
            
            # Create tabs for different visualizations
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Force Evolution", "Victory Analysis", "Domain Effectiveness", "Force Composition", "Survival Time"])
            
            with tab1:
                fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
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
                st.pyplot(fig1)
            
            with tab2:
                # Victory probability pie chart
                fig2, ax2 = plt.subplots(figsize=(8, 8))
                labels = ['Blue Victory', 'Red Victory', 'Draw']
                sizes = [blue_win_prob, red_win_prob, draw_prob]
                colors = ['blue', 'red', 'gray']
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax2.axis('equal')
                ax2.set_title("Victory Probability")
                
                st.pyplot(fig2)
                
                # Display victory probabilities
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Blue Victory Probability", f"{blue_win_prob:.1%}")
                with col2:
                    st.metric("Red Victory Probability", f"{red_win_prob:.1%}")
                with col3:
                    st.metric("Draw Probability", f"{draw_prob:.1%}")
            
            with tab3:
                # Domain effectiveness bar chart
                fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 6))
                
                # Blue domain effectiveness
                blue_domains = list(domain_effectiveness['blue'].keys())
                blue_values = list(domain_effectiveness['blue'].values())
                
                ax3a.bar(blue_domains, blue_values, color='blue', alpha=0.7)
                ax3a.set_title("Blue Force Domain Effectiveness")
                ax3a.set_xlabel("Domain")
                ax3a.set_ylabel("Fraction of Total Hits")
                ax3a.grid(True, axis='y')
                
                # Red domain effectiveness
                red_domains = list(domain_effectiveness['red'].keys())
                red_values = list(domain_effectiveness['red'].values())
                
                ax3b.bar(red_domains, red_values, color='red', alpha=0.7)
                ax3b.set_title("Red Force Domain Effectiveness")
                ax3b.set_xlabel("Domain")
                ax3b.set_ylabel("Fraction of Total Hits")
                ax3b.grid(True, axis='y')
                
                plt.tight_layout()
                st.pyplot(fig3)
                
                # Domain effectiveness table
                st.subheader("Domain Effectiveness Matrix")
                
                # Create a DataFrame for the effectiveness matrix
                effectiveness_data = []
                for i, attacker in enumerate(domains):
                    for j, defender in enumerate(domains):
                        blue_effectiveness = p_hit_blue[i,j] * (1 - p_int_red)
                        red_effectiveness = p_hit_red[i,j] * (1 - p_int_blue)
                        effectiveness_data.append({
                            'Attacker': attacker,
                            'Defender': defender,
                            'Blue Effectiveness': f"{blue_effectiveness:.2f}",
                            'Red Effectiveness': f"{red_effectiveness:.2f}"
                        })
                
                effectiveness_df = pd.DataFrame(effectiveness_data)
                st.dataframe(effectiveness_df)
            
            with tab4:
                # Force composition analysis
                st.subheader("Force Composition Analysis")
                
                # Calculate optimal force composition based on effectiveness
                blue_optimal = np.zeros(3)
                red_optimal = np.zeros(3)
                
                for i in range(3):
                    blue_optimal[i] = np.mean(p_hit_blue[i,:]) * (1 - p_int_red)
                    red_optimal[i] = np.mean(p_hit_red[i,:]) * (1 - p_int_blue)
                
                # Normalize to sum to 1
                blue_optimal = blue_optimal / np.sum(blue_optimal)
                red_optimal = red_optimal / np.sum(red_optimal)
                
                # Plot optimal force composition
                fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 6))
                
                ax4a.pie(blue_optimal, labels=domains, autopct='%1.1f%%', colors=['lightblue', 'blue', 'darkblue'])
                ax4a.set_title("Optimal Blue Force Composition")
                
                ax4b.pie(red_optimal, labels=domains, autopct='%1.1f%%', colors=['pink', 'red', 'darkred'])
                ax4b.set_title("Optimal Red Force Composition")
                
                plt.tight_layout()
                st.pyplot(fig4)
                
                # Compare current vs optimal composition
                blue_current = np.array(blue_initial) / np.sum(blue_initial)
                red_current = np.array(red_initial) / np.sum(red_initial)
                
                # Create a DataFrame for the comparison
                composition_data = []
                for i, domain in enumerate(domains):
                    composition_data.append({
                        'Domain': domain,
                        'Blue Current': f"{blue_current[i]:.2f}",
                        'Blue Optimal': f"{blue_optimal[i]:.2f}",
                        'Red Current': f"{red_current[i]:.2f}",
                        'Red Optimal': f"{red_optimal[i]:.2f}"
                    })
                
                composition_df = pd.DataFrame(composition_data)
                st.dataframe(composition_df)
            
            with tab5:
                # Survival time histogram
                fig5, ax5 = plt.subplots(figsize=(10, 6))
                ax5.hist(final_rounds, bins=20, alpha=0.7, color='green')
                ax5.axvline(mean_survival_time, color='red', linestyle='dashed', linewidth=2, 
                           label=f'Mean: {mean_survival_time:.1f} rounds')
                ax5.set_title("Distribution of Combat Duration")
                ax5.set_xlabel("Number of Rounds")
                ax5.set_ylabel("Frequency")
                ax5.grid(True)
                ax5.legend()
                
                st.pyplot(fig5)
                
                # Display survival time statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Mean Survival Time", f"{mean_survival_time:.1f} rounds")
                with col2:
                    st.metric("Std Dev of Survival Time", f"{std_survival_time:.1f} rounds")
            
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
            
            # Scenario comparison
            st.subheader("Scenario Comparison")
            st.write("Compare different scenarios by running multiple simulations with different parameters.")
            
            # Create a simple scenario comparison
            scenario_params = [
                ("Baseline", avg_shots_blue, avg_shots_red, p_int_blue, p_int_red),
                ("Increased Blue Firepower", avg_shots_blue * 1.5, avg_shots_red, p_int_blue, p_int_red),
                ("Increased Red Firepower", avg_shots_blue, avg_shots_red * 1.5, p_int_blue, p_int_red),
                ("Increased Blue Defense", avg_shots_blue, avg_shots_red, p_int_blue * 1.5, p_int_red),
                ("Increased Red Defense", avg_shots_blue, avg_shots_red, p_int_blue, p_int_red * 1.5)
            ]
            
            scenario_results = []
            
            for scenario_name, b_shots, r_shots, b_int, r_int in scenario_params:
                # Run a simplified simulation for this scenario
                blue_wins = 0
                red_wins = 0
                draw = 0
                
                for sim in range(50):  # Use fewer simulations for scenario comparison
                    blue = np.array(blue_initial)
                    red = np.array(red_initial)
                    
                    for round in range(max_rounds):
                        # Check for morale collapse
                        blue_ratio = np.sum(blue) / np.sum(blue_initial)
                        red_ratio = np.sum(red) / np.sum(red_initial)
                        
                        if blue_ratio < morale_threshold and np.random.random() < morale_collapse_chance:
                            blue = np.zeros_like(blue)
                        if red_ratio < morale_threshold and np.random.random() < morale_collapse_chance:
                            red = np.zeros_like(red)
                        
                        # Calculate shots (Poisson distribution)
                        blue_shots = np.random.poisson(b_shots * blue)
                        red_shots = np.random.poisson(r_shots * red)
                        
                        # Calculate hits (Binomial distribution)
                        blue_hits = np.zeros_like(blue)
                        red_hits = np.zeros_like(red)
                        
                        for i in range(3):  # Attacker domain
                            for j in range(3):  # Defender domain
                                if blue[i] > 0:
                                    p_hit = p_hit_blue[i,j] * (1 - r_int)
                                    hits = np.random.binomial(blue_shots[i], p_hit)
                                    blue_hits[j] += hits
                                if red[i] > 0:
                                    p_hit = p_hit_red[i,j] * (1 - b_int)
                                    hits = np.random.binomial(red_shots[i], p_hit)
                                    red_hits[j] += hits
                        
                        # Update forces
                        blue = np.maximum(0, blue - red_hits)
                        red = np.maximum(0, red - blue_hits)
                        
                        # Stop if one side is eliminated
                        if np.sum(blue) == 0 or np.sum(red) == 0:
                            break
                    
                    # Determine winner
                    if np.sum(blue) > 0 and np.sum(red) == 0:
                        blue_wins += 1
                    elif np.sum(red) > 0 and np.sum(blue) == 0:
                        red_wins += 1
                    else:
                        draw += 1
                
                # Calculate victory probabilities
                blue_win_prob = blue_wins / 50
                red_win_prob = red_wins / 50
                draw_prob = draw / 50
                
                scenario_results.append({
                    'Scenario': scenario_name,
                    'Blue Victory': f"{blue_win_prob:.1%}",
                    'Red Victory': f"{red_win_prob:.1%}",
                    'Draw': f"{draw_prob:.1%}"
                })
            
            # Display scenario comparison
            scenario_df = pd.DataFrame(scenario_results)
            st.dataframe(scenario_df)
            
            # Plot scenario comparison
            fig6, ax6 = plt.subplots(figsize=(12, 6))
            
            scenarios = [r['Scenario'] for r in scenario_results]
            blue_probs = [float(r['Blue Victory'].strip('%')) / 100 for r in scenario_results]
            red_probs = [float(r['Red Victory'].strip('%')) / 100 for r in scenario_results]
            draw_probs = [float(r['Draw'].strip('%')) / 100 for r in scenario_results]
            
            x = np.arange(len(scenarios))
            width = 0.25
            
            ax6.bar(x - width, blue_probs, width, label='Blue Victory', color='blue')
            ax6.bar(x, red_probs, width, label='Red Victory', color='red')
            ax6.bar(x + width, draw_probs, width, label='Draw', color='gray')
            
            ax6.set_title("Scenario Comparison")
            ax6.set_xlabel("Scenario")
            ax6.set_ylabel("Probability")
            ax6.set_xticks(x)
            ax6.set_xticklabels(scenarios, rotation=45, ha='right')
            ax6.legend()
            ax6.grid(True, axis='y')
            
            plt.tight_layout()
            st.pyplot(fig6)

if __name__ == "__main__":
    main() 
