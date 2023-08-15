#===================================================================================================
#?                                           ABOUT
#  @author         :  Alfonso Toriz Vazquez
#  @email          :  atoriz98@comunidad.unam.mx
#  @createdOn      :  02082023
#  @description    :  Simple static system estimator. We want to estimate the weight of a gold bar
#                     using an unbiased scale. The system is the gold bar and the system state is
#                     its weight. Based on the book Kalman Filter from the Ground up by Alex Becker.
#===================================================================================================

#===================================================================================================
#  *                                        MATH NOTATION 
#
#    x                 = true value of the weight.
#    z_n               = measured value of the weight at time n.
#    \hat{x}_{n,n}     = estimate of the x at time n, made after taking the measurement z_n.
#    \hat{x}_{n+1,n}   = estimate of the future state (n+1) of x. Made at time n. It is a predicted 
#                        state.
#    \hat{x}_{n-1,n-1} = estimate of x at time n - 1. Made after taking the measurement  z_{n-1}.
#    \hat{x}_{n,n-1}   = prior prediction minus estimate of the state at time n. The prediction is 
#                        made at time n - 1.
#
#   The \hat over a variable indicates an estimated value.
#
#!  It is a static/constant system, that means -> \hat{x}_{n+1, n} = \hat{x}_{n,n} that is the
#!  future state or predicted state equals the estimated state since it doesn't change over time. 
#===================================================================================================

#===================================================================================================
#?                                          IMPORTS
#===================================================================================================

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#======================================= END OF IMPORTS ============================================

#===================================================================================================
#?                                         CONSTANTS
#===================================================================================================

REAL_BAR_WEIGHT = 1_000.0
INITIAL_GUESS = 1_000.0  # Needed for filter initialization
MEASUREMENTS = [996.0, 994.0, 1_021.0, 1_000.0, 1_002.0, 1_010.0, 983.0, 971.0, 993.0, 1_023.0]  # page 48

# Uncomment following lines to change values to random initialization
# ERROR = 1
# NUMBER_OF_MEASUREMENTS = 10
# RNG = np.random.default_rng(seed=42)

# MEASUREMENTS = RNG.uniform(
#     low=REAL_BAR_WEIGHT - ERROR,
#     high=REAL_BAR_WEIGHT + ERROR,
#     size=NUMBER_OF_MEASUREMENTS)  # Scale's measurements

#====================================== END OF CONSTANTS ===========================================

#===================================================================================================
#?                                            FUNCTIONS 
#===================================================================================================

def naive_estimation(measurements: list[float]) -> float:
    """Returns an estimation based on previous measurements. It uses a naive and inefficient
    approach, the average of all previous data.

    \hat{x}_{n,n} = \frac{1}{n}(z_1, z_2, ..., z_{n-1}, z_{n}) = \frac{1}{n} \sum_{i=1}^n (z_i)
    Eq. 3.1 

    Args:
        measurements (list[floats]): list of measurements 

    Returns:
        average (float): average of all previous measurements 
    """
    average = np.average(measurements)
    return average



def state_update_equation(predicted_value_of_current_state: float, kalman_gain: float, current_measurement: float) -> float:
    """Obtains an estimate using the State Update Equation (Eq 3.3):


    Much better than the naive approach since it doesn't take into account all previous
    measurements. Works wonders for static state systems, such as the weight of an object.

    Args:
        predicted_value_of_current_state (float): self explanatory 
        kalman_gain (float): factor term that determines the weight of new measurements
        current_measurement (float): self explanatory 

    Returns:
        estimate_current_state (float): current state estimation based on past and present information
    """
    # Equation 3.3:
    #    \hat{x}_{n,n}     =          \{x}_{n, n-1}           +  \alpha_n   * (       z_n          -         \hat{x}_{n, n-1}        )
    estimate_current_state = predicted_value_of_current_state + kalman_gain * (current_measurement - predicted_value_of_current_state)
    return estimate_current_state

#========================================= END OF FUNCTIONS ========================================

#===================================================================================================
#?                                              MAIN 
#===================================================================================================

if __name__ == "__main__":
    #===========================
    #?           UI 
    #===========================
    print("\nSimple static system estimator\n")
    print(f"Real bar weight: {REAL_BAR_WEIGHT}")
    print(f"Measurements: ")
    print(*[str(measurement) + " g " for measurement in MEASUREMENTS])
    
    #===========================
    #?       ESTIMATIONS 
    #===========================
    # Naive Method
    average_estimation = naive_estimation(MEASUREMENTS)

    # State Update Equation Method
    state_estimates_list = [INITIAL_GUESS]  # First value needed for initialization
    kalman_gain_list = []
    
    # Computes estimated states for each measurement
    for idx, measurement in enumerate(MEASUREMENTS):
        # print(f"Measurement {idx}: {MEASUREMENTS[idx]}")
    
        # Compute Kalman gain
        kalman_gain_list.append(1 / len(state_estimates_list))
        # Compute current estimate state
        current_estimate_state = state_update_equation(
            predicted_value_of_current_state=state_estimates_list[idx],
            kalman_gain=kalman_gain_list[idx],
            current_measurement=measurement)
        
        # print(f"Current estimate: {current_estimate_state}")
        state_estimates_list.append(current_estimate_state)
    
    # Printing results...
    print("\nWeight estimations:")
    print(f"Naive estimation: {naive_estimation(MEASUREMENTS):0.2f} g")
    print(f"State update equation estimation: {state_estimates_list[-1]:0.2f} g\n")
    
    #===========================
    #?     DATA PROCESSING 
    #===========================
    # Since the dynamic model is static:
    state_predictions_list = state_estimates_list[1:].copy()  # Shallow copy since we will not modify its elements
    
    # Sanity check
    assert(len(kalman_gain_list) == len(MEASUREMENTS) == (len(state_estimates_list) - 1) == len(state_predictions_list))
    
    data = np.column_stack(
        [
            [0, *kalman_gain_list],
            [0, *MEASUREMENTS],
            state_estimates_list,
            [0, *state_predictions_list]
        ]
    )  # We insert zeros to account for the additional value, INITIAL_GUESS, in the estimates list to keep dimensions equal
    
    data_pd = pd.DataFrame(
        data=data,
        columns=[
            "Kalman Gain",
            "Measurements",
            "Estimates",
            "Predictions"
        ]
    )
    
    data_pd.rename_axis("Iterations", inplace=True)
    data_pd.to_csv("example_1_static_system_state_update_equation_estimation_data.csv")
    print(data_pd)
    
    #===========================
    #?         PLOTTING 
    #===========================
    x = [i for i in range(len(MEASUREMENTS))]  # Same x range for every plot
    
    # GOLD BAR WEIGHT PLOT
    plt.figure(figsize=(10, 5))
    sns.scatterplot(
        x=x,
        y=MEASUREMENTS,
        label="Measurements",
        color="red",
        s=50
    )  # Scale's measurements
    
    sns.lineplot(
        x=x,
        y=[REAL_BAR_WEIGHT for i in range(len(MEASUREMENTS))],
        linestyle="dashed",
        label="Real value",
        color="green"
    )  # Constant value
    
    # Plot config
    plt.title("Gold bar weight")
    plt.xlabel("Measurements")
    plt.ylabel("Weight (g)")
    plt.ticklabel_format(style="plain", useOffset=False)  # Disable scientific notation
    plt.xticks(ticks=[i for i in range(0,10)], labels=[i for i in range(1,11)])
    plt.savefig("gold_bar_weight_plot.png", dpi=300)  # Uncomment to save figure
    # plt.show()
    
    
    # RESULT ANALYSIS PLOT
    plt.figure(figsize=(10,5))
    sns.lineplot(
        x=x,
        y=MEASUREMENTS,
        marker="o",
        c="blue",
        label="Measurements"
    )
    sns.lineplot(
        x=x,
        y=[REAL_BAR_WEIGHT for i in range(len(MEASUREMENTS))],
        marker="v",
        c="green",
        label="True values",
        linestyle="dashed"
    )
    sns.lineplot(
        x=x,
        y=state_estimates_list[1:],
        marker="D",
        c="red",
        label="Estimates"
    )
    
    # Plot config
    plt.title("Result Analysis")
    plt.xlabel("Iterations")
    plt.ylabel("Weight (g)")
    plt.ticklabel_format(style="plain", useOffset=False)  # Disable scientific notation
    plt.xticks(ticks=[i for i in range(0,10)], labels=[i for i in range(1,11)])
    plt.savefig("result_analysis.png", dpi=300)  # Uncomment to save figure
    # plt.show()

#========================================== END OF MAIN ============================================