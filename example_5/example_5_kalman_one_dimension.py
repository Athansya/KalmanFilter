'''
File: example_5_kalman_one_dimension.py
Project: example_5
File Created: Thursday, 31st August 2023 1:23:08 pm
Author: Alfonso Toriz Vazquez (atoriz98@comunidad.unam.mx)
-----
Last Modified: Thursday, 31st August 2023 1:23:30 pm
Modified By: Alfonso Toriz Vazquez (atoriz98@comunidad.unam.mx>)
-----
License: MIT License
-----
Description: Building height estimation using an unidimensional
Kalman filter. It is a constant dynamic model, since the height
does not change.
'''

# ---------------------------------------------------------------------------- #
#                                    IMPORTS                                   #
# ---------------------------------------------------------------------------- #
import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------- #
#                        MATPLOTLIB ENABLE LATEX FIGURES                       #
# ---------------------------------------------------------------------------- #
rc("font", **{"family": "serif", "serif": ["Computer Modern"], "size": 16})
rc("text", usetex=True)

# ---------------------------------------------------------------------------- #
#                                   CONSTANTS                                  #
# ---------------------------------------------------------------------------- #
REAL_HEIGHT = 50.0  # mts 
MEASUREMENT_ERROR = 5.0  # or std in meters

MEASUREMENTS = [
    49.03,
    48.44,
    55.21,
    49.98,
    50.60,
    52.61,
    45.87,
    42.64,
    48.26,
    55.84
]  # page 90

# Human's estimation
INITIAL_HEIGHT_ESTIMATION = 60.0  # mts
INITIAL_VARIANCE_ESTIMATION = 225.0  # or sigma^2 = 15^2 = 225


# ---------------------------------------------------------------------------- #
#                            EXTRAPOLATION EQUATIONS                           #
# ---------------------------------------------------------------------------- #

def state_extrapolation_equation(current_estimation: float) -> float:
    return current_estimation  # prediction

def variance_extrapolation_equation(current_variance: float) -> float:
    return current_variance  # prediction

# ---------------------------------------------------------------------------- #
#                               UPDATE EQUATIONS                               #
# ---------------------------------------------------------------------------- #

def state_update_equation(
    predicted_estimation: float,
    kalman_gain: float,
    current_measurement: float
) -> float:
    """State update equation based on the current prediction,
    Kalman gain and measurement.

    Args:
        predicted_estimation (float): predicted estimation from
        the state extrapolation equation. 
        kalman_gain (float): kalman gain from its equation. 
        current_measurement (float): current measurement from sensor. 

    Returns:
        current_estimation (float): current state estimation.
    """

    current_estimation = (
        (1 - kalman_gain) * predicted_estimation +
        kalman_gain * current_measurement
    )
    return current_estimation 

def variance_update_equation(
    kalman_gain: float,
    predicted_variance: float
) -> float:
    """Variance update equation based on current Kalman gain
    and the current predicted variance.

    Args:
        kalman_gain (float): current kalman gain. 
        predicted_variance (float): predicted variance from
        the variance extrapolation equation. 

    Returns:
        variance_estimation (float): current variance estimation. 
    """
    variance_estimation = (1 - kalman_gain) * predicted_variance
    return variance_estimation

def kalman_gain_equation(
   predicted_variance: float,
   measurement_variance: float
) -> float:
    """Factor used by the state update equations.

    Args:
        predicted_variance (float): current predicted variance. 
        measurement_variance (float): measurement uncertainty. 

    Returns:
        current_kalman_gain (float): current value for the
        Kalman gain equation. 
    """
    current_kalman_gain = (
        predicted_variance / (predicted_variance + measurement_variance)
    )
    return current_kalman_gain



def main():
    # Lists of values
    # Predictions
    predicted_states = [
        state_extrapolation_equation(INITIAL_HEIGHT_ESTIMATION)
    ]
    predicted_variances = [
        variance_extrapolation_equation(INITIAL_VARIANCE_ESTIMATION)
    ]

    # Kalman gain
    kalman_gain_values = []

    # States
    estimated_states = []
    estimated_variances = []

    # Algorithm loop
    for index, z_n in enumerate(MEASUREMENTS):
        print(f"Iteration {index}")
        # Update
        kalman_gain = kalman_gain_equation(
            predicted_variance=predicted_variances[index],
            measurement_variance=MEASUREMENT_ERROR,
        )

        current_state_estimation = state_update_equation(
            predicted_estimation=predicted_states[index],
            kalman_gain=kalman_gain,
            current_measurement=z_n
        )

        current_variance_estimation = variance_update_equation(
            kalman_gain=kalman_gain,
            predicted_variance=predicted_variances[index]
        )

        kalman_gain_values.append(kalman_gain)
        estimated_states.append(current_state_estimation)
        estimated_variances.append(current_variance_estimation)

        # Prediction
        predicted_state = state_extrapolation_equation(
            current_estimation=current_state_estimation
        )

        predicted_variance = variance_extrapolation_equation(
            current_variance=current_variance_estimation
        )

        # Saves values
        predicted_states.append(predicted_state)
        predicted_variances.append(predicted_variance)

        #TODO CREATE PANDAS DATAFRAME WITH VALUES IN ORDER
    # Dataframe creation
    # Creates data table
    data = np.column_stack(
        [
            [0, *MEASUREMENTS],
            [0, *estimated_states],
            [0, *estimated_variances],
            [0, *kalman_gain_values],
            predicted_states,
            predicted_variances,
        ]
    )

    data_pd = pd.DataFrame(
        data=data,
        columns=[
            "Measurements",
            "Estimated Height",
            "Estimated Variance",
            "Kalman Gain",
            "Predicted Height",
            "Predicted Variance"
        ]
    )

    data_pd.rename_axis("Iterations", inplace=True)

    data_pd.to_csv("data.csv")
    print(data_pd)

    #TODO CREATE PLOTS
    # Kalman Gain Plot
    x_range = [i for i in range(len(MEASUREMENTS))]
    sns.lineplot(
        x=x_range,
        y=data_pd["Kalman Gain"][1:],
        marker="o"
    ).set_title("Kalman Gain")
    plt.xlabel("Measurement number")
    plt.savefig("Kalman_gain_values.png")

    # Result Analysis
    fig, ax = plt.subplots(figsize=(10,5), layout='tight')
    sns.lineplot(
        ax=ax,
        x=x_range,
        y=data_pd["Measurements"][1:],
        label="Measurements",
        color="blue",
        marker='s'
    )
    sns.lineplot(
        ax=ax,
        x=x_range,
        y=data_pd["Estimated Height"][1:],
        label="Estimations",
        color="red",
        marker='o'
    )
    sns.lineplot(
        ax=ax,
        x=x_range,
        y=data_pd["Predicted Height"][:-1],
       label="Predictions",
       color="gray",
       marker="D"
    )
    sns.lineplot(
        ax=ax, 
        x=x_range,
        y=[REAL_HEIGHT for _ in range(len(x_range))],
        label="True Values",
        color="green",
       marker="^"
        )
    sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
    plt.title("Building Height")
    plt.xlabel("Measurement number")
    plt.ylabel("Height (m)")
    plt.savefig("Building Height")
        

if __name__ == "__main__":
    main()