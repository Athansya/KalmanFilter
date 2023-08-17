"""
File: example_2_alpha_beta_state_extrapolation_equation.py
Project: Kalman Filter 
File Created: Tuesday, 15th August 2023 11:22:09 pm
Author: Alfonso Toriz Vazquez (atoriz98@comunidad.unam.mx)
-----
Last Modified: Tuesday, 15th August 2023 11:45:27 pm
Modified By: Alfonso Toriz Vazquez (atoriz98@comunidad.unam.mx>)
-----
License: MIT License
-----
Description: Estimates an aircraft position and velocity using the state
extrapolation equations and the alpha beta filters. Based on the example
2 from Alex Becker's book, Kalman Filter from the Ground up.
"""

# ---------------------------------------------------------------------------- #
# NOTE                          PROBLEM DESCRIPTION                            #
# ---------------------------------------------------------------------------- #
#                                                                              #
#        Imagine an aircraft flying at a constant altitude and velocity.       #
#    We want to estimate its position from a radar as well as its velocity.    #
#    For the sake of the example the velocity is obtained from the derivate    #
#    of the position. The radar tracks the position of the aircraft every 5    #
#    seconds, known as delta t.                                                #
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
INITIAL_ESTIMATED_POSITION = 30_000  # Meters
INITIAL_ESTIMATED_VELOCITY = 40  # m/s
MEASUREMENTS = [
    30171,
    30353,
    30756,
    30799,
    31018,
    31278,
    31276,
    31379,
    31748,
    32175
]  # Sensor measurements
ALPHA = 0.2  # Filter param
BETA = 0.1  # Filter param


# ---------------------------------------------------------------------------- #
#                         STATE EXTRAPOLATION EQUATION                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
#                Position -> x_{n+1} = x_n + \DeLta t \dot{x}_n                #
#                Velocity -> \dot{x}_{n+1} = \dot{x}_n  (Constant)             #
#                                                                              #
# ---------------------------------------------------------------------------- #
def state_extrapolation_first_equation(
    current_state: float,
    delta: float,
    current_state_derivative: float
) -> float:
    """Obtains predicted state based on current state. Uses the initial
    conditions to extrapolate the value of the first iteration of the
    algorithm.

    Args:
        current_state (float): current state based on initial conditions.
        delta (float): track-to-track period in seconds $\Delta t$.
        current_state_derivative (float): current state derivative based on
        initial conditions.

    Returns:
        predicted_state (float): predicted state extrapolated from initial
        conditions.
    """
    predicted_state = current_state + delta * current_state_derivative
    return predicted_state 

def state_extrapolation_second_equation(current_state_derivative: float) -> float:
    """ 

    Args:
        current_state_derivative (float): current state derivate based on
        initial conditions.

    Returns:
        predicted_state (float): predicted state extrapolated from initial
        conditions.
    """
    predicted_state_derivate = current_state_derivative  # Constant!!!
    return predicted_state_derivate
# ---------------------------------------------------------------------------- #
#                              ALPHA - BETA FILTER                             #
# ---------------------------------------------------------------------------- #
#                                                                              #
#    Filter that assumes that in system with two internal states the second    #
#    state is obtained by the derivate of the first one. A great example is    #
#     using position's derivative to compute velocity. Its values depend on    #
#                      the precision of the measurements.                      #
#                                                                              #
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
# ---------------------------------------------------------------------------- #
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

# ---------------------------------------------------------------------------- #
#                                     MAIN                                     #
# ---------------------------------------------------------------------------- #

if __name__ == "__main__":
    # Lists of predicted values
    predicted_positions_list = [
        state_extrapolation_first_equation(
            current_state=INITIAL_ESTIMATED_POSITION,
            delta=DELTA_T,
            current_state_derivative=INITIAL_ESTIMATED_VELOCITY
        )
    ]

    predicted_velocities_list = [
        state_extrapolation_second_equation(
            current_state_derivative=INITIAL_ESTIMATED_VELOCITY
        )
    ]

    # Lists of estimated values
    estimated_positions_list = []
    estimated_velocities_list = []

    # Number of iterations
    N_ITERATIONS = 10

    # Algorithm cycle
    for i in range(N_ITERATIONS):

        # Obtain current estimations based on predictions, filter and measures
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

        estimated_positions_list.append(estimated_position_state)
        estimated_velocities_list.append(estimated_velocity_state)

        # Obtain predictions
        predicted_position_state = state_extrapolation_first_equation(
            current_state=estimated_position_state,
            delta=DELTA_T,
            current_state_derivative=estimated_velocity_state
        )

        predicted_velocity_state = state_extrapolation_second_equation(
            current_state_derivative=estimated_velocity_state
        )

        predicted_positions_list.append(predicted_position_state)
        predicted_velocities_list.append(predicted_velocity_state)

    # Creates data table 
    data = np.column_stack(
        [
            [0, *MEASUREMENTS],
            [0, *estimated_positions_list],
            [0, *estimated_velocities_list],
            predicted_positions_list,
            predicted_velocities_list
        ]
    )

    data_pd = pd.DataFrame(
        data=data,
        columns=[
            "Measurements",
            "Estimated Position",
            "Estimated Velocity",
            "Predicted Position",
            "Predicted Velocity"
        ]
    )
    data_pd.rename_axis("Iterations", inplace=True)

    data_pd.to_csv("data.csv")  # Saves information
    print(data_pd)  # Shows pandas DataFrame

    # Plots
    time = [DELTA_T * i for i in range(len(MEASUREMENTS))]  # Each measurement was made each DELTA_T seconds

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
        y=data_pd["Predicted Position"][1:],
        c="gray",
        marker="^",
        label="Prediction"
    )

    plt.title(f"Position vs. Time ($\\alpha =$ {ALPHA}, $\\beta =$ {BETA})")
    plt.xlabel("Time (s)")
    plt.ylabel("Position (m)")

    # plt.savefig("Results_analysis.png", dpi=300)