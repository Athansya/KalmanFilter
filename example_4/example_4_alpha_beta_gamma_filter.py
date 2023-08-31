'''
File: example_3_alpha_beta_gamma_filter.py
Project: example_4
File Created: Tuesday, 22nd August 2023 10:03:25 pm
Author: Alfonso Toriz Vazquez (atoriz98@comunidad.unam.mx)
-----
Last Modified: Tuesday, 22nd August 2023 10:03:37 pm
Modified By: Alfonso Toriz Vazquez (atoriz98@comunidad.unam.mx>)
-----
License: MIT License
-----
Description: Estimates an aircraft position, velocity and acceleration
using the state extrapolation and state update equations along with an
alpha beta gamma filter.
'''

# ---------------------------------------------------------------------------- #
#                               PROBLEM DESCRIPTION                            #
# ---------------------------------------------------------------------------- #
#                                                                              #
#      Now we will track an aircraft that moves at a constant velocity of      #
#      50 m/s for 20 sec. Then it accelerates with a constant 8 m/s^2 for      #
#                                  35 seconds.                                 #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                                   LIBRARIES                                  #
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import seaborn as sns
import pandas as pd

# ---------------------------------------------------------------------------- #
#                        MATPLOTLIB ENABLE LATEX FIGURES                       #
# ---------------------------------------------------------------------------- #
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 16})
rc("text", usetex=True)

# ---------------------------------------------------------------------------- #
#                                   CONSTANTS                                  #
# ---------------------------------------------------------------------------- #
DELTA_T = 5  # Seconds, track-to-track period
INITIAL_ESTIMATED_POSITION = 30_000  # meters
INITIAL_ESTIMATED_VELOCITY = 50  # m/s
INITIAL_ESTIMATED_ACCELERATION = 0  # m/s^2
MEASUREMENTS = [
    30221,
    30453,
    30906,
    30999,
    31368,
    31978,
    32526,
    33379,
    34698,
    36275
]  # Sensor measurements
ALPHA = 0.5  # Filter param
BETA = 0.4  # Filter param
GAMMA = 0.1

# ---------------------------------------------------------------------------- #
#                         STATE EXTRAPOLATION EQUATION                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
#              Position -> x_{n+1} = x_n + \dot{x}_n Delta t + ...             #
#                     ... + \ddot{x}_n \frac{\Delta t^2}{2}                    #
#                                                                              #
#          Velocity -> \dot{x}_{n+1} = \dot{x}_n + \ddot{x}_n \Delta t         #
#                                                                              #
#            Acceleration -> \ddot{x}_{n+1} = \ddot{x}_n (Constant)            #
#                                                                              #
# ---------------------------------------------------------------------------- #
def state_extrapolation_first_equation(
    current_state: float,
    delta: float,
    current_state_derivative: float,
    current_state_second_derivative: float
) -> float:
    """Obtains predicted state based on current state. Uses the initial
    conditions to extrapolate the value of the first iteration of the
    algorithm.

    Args:
        current_state (float): current state based on initial conditions.
        delta (float): track-to-track period in seconds $\Delta t$.
        current_state_derivative (float): current state derivative based on
        initial conditions.
        current_state_second_derivative (float): current state second
        derivative based on initial conditions

    Returns:
        predicted_state (float): predicted state extrapolated from initial
        conditions.
    """
    predicted_state = current_state + current_state_derivative * delta + (
        current_state_second_derivative * (delta**2) / 2)
    return predicted_state 

def state_extrapolation_second_equation(
    current_state_derivative: float,
    current_state_second_derivative: float,
    delta: float
) -> float:
    """Obtains predicted state based on the current state.

    Args:
        current_state_derivative (float): current state derivate based on
        initial conditions.
        current_state_second_derivative (float): current state second
        derivative based on initial conditions

    Returns:
        predicted_state (float): predicted state extrapolated from initial
        conditions.
    """
    predicted_state_derivate = current_state_derivative + (
        current_state_second_derivative * delta)
    return predicted_state_derivate


def state_extrapolation_third_equation(
    current_state_second_derivative: float
) -> float:
    """Obtains predicted state based on the current state.

    Args:
        current_state_second_derivative (float): current state second
        derivative based on initial conditions.

    Returns:
        predicted_state_second_derivative (float): predicted state
        extrapolated from current one. 
    """
    # Constant!
    predicted_state_second_derivative = current_state_second_derivative  
    return predicted_state_second_derivative


# ---------------------------------------------------------------------------- #
#                            STATE UPDATE EQUATIONS                            #
# ---------------------------------------------------------------------------- #
#                                                                              #
#              Position -> \hat{x}_{n, n} = \hat{x}_{n, n-1} + ...             #
#                     ... + \alpha (z_n - \hat{x}_{n, n-1})                    #
#                                                                              #
#        Velocity -> \hat{\dot{x}}_{n, n} = \hat{\dot{x}}_{n, n-1} + ...       #
#            ... + \beta (\frac{z_n  - \hat{x}_{n, n-1}}{\Delta t})            #
#                                                                              #
#    Acceleration -> \hat{\ddot{x}}_{n, n} = \hat{\ddot{x}}_{n , n-1} + ...    #
#         ... + \gamma (\frac{z_n - \hat{x}_{n, n-1}}{0.5 \Delta t^2})         #
#                                                                              #
# ---------------------------------------------------------------------------- #
def state_update_first_equation(
    current_state_prediction: float,
    alpha: float,
    current_measurement: float,
) -> float:
    """Obtains the current estimate value based on the state update
    equations and an alpha-beta filter. The alpha value depends on the sensor
    precision. This equation is obtained by integrating the second one.

    Args:
        current_state_prediction (float): predicted value of the current 
        state $x_{n, n}$ based on the previous information $x_{n, n-1}$.
        beta (float): alpha-beta filter argument.
        current_measurement (float): current sensor measurement $z_n$.

    Returns:
        (float): current state estimation based on past and 
        present information.
    """
    current_state_estimate = current_state_prediction + alpha * (
        current_measurement - current_state_prediction
    )
    return current_state_estimate 


def state_update_second_equation(
    current_state_prediction_derivative: float,
    beta: float,
    current_state_prediction: float,
    current_measurement: float,
    delta: float,
) -> float:
    """Obtains the current estimate value based on the state update 
    equations and an alpha-beta filter. The beta value depends on the sensor
    precision. This equation is obtained by derivating the first one.

    Args:
        current_state_prediction_derivative (float): predicted value of 
        the current state derivative $\dot{x}_{n, n}$ based on the previous
        information $\dot{x}_{n, n-1}$.
        current_state_prediction (float): predicted value of the current state
        $x_{n, n}$ based on the previous information $x_{n, n-1}$.
        beta (float): alpha-beta filter argument.
        current_measurement (float): current sensor measurement $z_n$.
        delta (float): track-to-track period in seconds $/Delta t$.

    Returns:
        current_state_estimate_derivative (float): current state derivative estimation
        based on past and present information.
    """
    current_state_estimate_derivative = current_state_prediction_derivative + beta * (
        (current_measurement - current_state_prediction) / delta
    )
    return current_state_estimate_derivative


def state_update_third_equation(
    current_state_prediction_second_derivative: float,
    gamma: float,
    current_state_prediction: float,
    current_measurement: float,
    delta: float
) -> float:
    """_summary_

    Args:
        current_state_prediction_second_derivative (float): predicted value of 
        the current state second derivative $\ddot{x}_{n, n}$ based on the previous
        information $\ddot{x}_{n, n-1}$.
        gamma (float): alpha-beta-gamma filter argument.
        current_state_prediction (float): predicted value of the current state
        $x_{n, n}$ based on the previous information $x_{n, n-1}$.
        current_measurement (float): current sensor measurement $z_n$.
        delta (float): track-to-track period in seconds $/Delta t$.


    Returns:
        current_state_estimate_second_derivative (float): current state second
        derivative estimation based on past and present information 
    """
    current_state_estimate_second_derivative = (
        current_state_prediction_second_derivative + gamma * (
            (current_measurement - current_state_prediction) /
            (0.5 * delta**2)
        )
    )
    return current_state_estimate_second_derivative


# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    # List of predicted values
    predicted_positions_list = [
        state_extrapolation_first_equation(
            current_state=INITIAL_ESTIMATED_POSITION,
            delta=DELTA_T,
            current_state_derivative=INITIAL_ESTIMATED_VELOCITY,
            current_state_second_derivative=INITIAL_ESTIMATED_ACCELERATION
        )
    ]

    predicted_velocities_list = [
        state_extrapolation_second_equation(
            current_state_derivative=INITIAL_ESTIMATED_VELOCITY,
            current_state_second_derivative=INITIAL_ESTIMATED_ACCELERATION,
            delta=DELTA_T
        )
    ]

    predicted_accelerations_list = [
        state_extrapolation_third_equation(
            current_state_second_derivative=INITIAL_ESTIMATED_ACCELERATION
        )
    ]

    # Lists of estimated values
    estimated_positions_list = []
    estimated_velocities_list = []
    estimated_accelerations_list = []

    # Number of iterations
    N_ITERATIONS = 10

    # Alogrithm cycle
    for i in range(N_ITERATIONS):
        # Obtain current estimations based on predictions, filter and
        # measurements

        estimated_position_state = state_update_first_equation(
            current_state_prediction=predicted_positions_list[-1],
            alpha=ALPHA,
            current_measurement=MEASUREMENTS[i]
        )
        
        estimated_velocity_state = state_update_second_equation(
            current_state_prediction_derivative=predicted_velocities_list[-1],
            beta=BETA,
            current_state_prediction=predicted_positions_list[-1],
            current_measurement=MEASUREMENTS[i],
            delta=DELTA_T
        )

        estimated_acceleration_state = state_update_third_equation(
            current_state_prediction_second_derivative=predicted_accelerations_list[-1],
            gamma=GAMMA,
            current_state_prediction=predicted_positions_list[-1],
            current_measurement=MEASUREMENTS[i],
            delta=DELTA_T
        )


        estimated_positions_list.append(estimated_position_state)
        estimated_velocities_list.append(estimated_velocity_state)
        estimated_accelerations_list.append(estimated_acceleration_state)

        # Obtain predictions
        predicted_position_state = state_extrapolation_first_equation(
            current_state=estimated_position_state,
            delta=DELTA_T,
            current_state_derivative=estimated_velocity_state,
            current_state_second_derivative=estimated_acceleration_state
        )

        predicted_velocity_state = state_extrapolation_second_equation(
            current_state_derivative=estimated_velocity_state,
            current_state_second_derivative=estimated_acceleration_state,
            delta=DELTA_T
        )

        predicted_acceleration_state = state_extrapolation_third_equation(
            current_state_second_derivative=estimated_acceleration_state
        )

        predicted_positions_list.append(predicted_position_state)
        predicted_velocities_list.append(predicted_velocity_state)
        predicted_accelerations_list.append(predicted_acceleration_state)

    # Creates data table
    data = np.column_stack(
        [
            [0, *MEASUREMENTS],
            [0, *estimated_positions_list],
            [0, *estimated_velocities_list],
            [0, *estimated_accelerations_list],
            predicted_positions_list,
            predicted_velocities_list,
            predicted_accelerations_list
        ]
    )

    data_pd = pd.DataFrame(
        data=data,
        columns=[
            "Measurements",
            "Estimated Position",
            "Estimated Velocity",
            "Estimated Acceleration",
            "Predicted Position",
            "Predicted Velocity",
            "Predicted Acceleration"
        ]
    )
    data_pd.rename_axis("Iterations", inplace=True)

    data_pd.to_csv("data.csv")
    print(data_pd)

    # Plots
    time = [DELTA_T * i for i in range(len(MEASUREMENTS))]  # x-axis

    plt.figure(figsize=(15,10))
    sns.lineplot(
        x=time,
        y=data_pd["Measurements"][1:],
        c="blue",
        marker="s",
        label="Measurements"
    )
    sns.lineplot(
        x=time,
        y=data_pd["Estimated Position"][1:],
        c="red",
        marker="o",
        label="Estimates"
    )
    sns.lineplot(
        x=time,
        y=data_pd["Predicted Position"][:-1],
        c="gray",
        marker="^",
        label="Prediction"
    )

    plt.title(f"Position vs. Time ($\\alpha =$ {ALPHA}, $\\beta =$ {BETA}, $\\gamma =$ {GAMMA})")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")
    plt.savefig("Result_analysis_position.png", dpi=300)


    plt.figure(figsize=(15,10))
    sns.lineplot(
        x=time,
        y=data_pd["Estimated Velocity"][1:],
        c="red",
        marker="o",
        label="Estimates"
    )
    sns.lineplot(
        x=time,
        y=data_pd["Predicted Velocity"][:-1],
        c="gray",
        marker="^",
        label="Prediction"
    )

    plt.title(f"Velocity vs. Time ($\\alpha =$ {ALPHA}, $\\beta =$ {BETA}, $\\gamma =$ {GAMMA})")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/2)")
    plt.savefig("Result_analysis_velocity.png", dpi=300)

    plt.figure(figsize=(15,10))
    sns.lineplot(
        x=time,
        y=data_pd["Estimated Acceleration"][1:],
        c="red",
        marker="o",
        label="Estimates"
    )
    sns.lineplot(
        x=time,
        y=data_pd["Predicted Acceleration"][:-1],
        c="gray",
        marker="^",
        label="Prediction"
    )

    plt.title(f"Acceleration vs. Time ($\\alpha =$ {ALPHA}, $\\beta =$ {BETA}, $\\gamma =$ {GAMMA})")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration ($m/s^2$)")
    plt.savefig("Result_analysis_acceleration.png", dpi=300)