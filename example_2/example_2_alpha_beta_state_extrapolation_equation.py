'''
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
'''

# ---------------------------------------------------------------------------- #
#NOTE                          PROBLEM DESCRIPTION                             #
# ---------------------------------------------------------------------------- #
#                                                                              #
#        Imagine an aircraft flying at a constant altitude and velocity.       #
#    We want to estimate its position from a radar as well as its velocity.    #
#    For the sake of the example the velocity is obtained from the derivate    #
#    of the position. The radar tracks the position of the aircraft every 5    #
#    seconds, known as delta t.                                                #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                         STATE EXTRAPOLATION EQUATION                         #
# ---------------------------------------------------------------------------- #
#                                                                              #
#                Position -> x_{n+1} = x_n + \Delta t \dot{x}_n                #
#                Velocity -> \dot{x}_{n+1} = \dot{x}_n  (Constant)             #
#                                                                              #
# ---------------------------------------------------------------------------- #
#                              ALPHA - BETA FILTER                             #
# ---------------------------------------------------------------------------- #
#                                                                              #
#       Filter that assumes that given a system with two internal states,      #
#       the second state is obtained by the derivative of the first one.       #
#     A great example is using position's derivative to obtain its velocity    #
#            Its values depend on the precision of the measurements            #
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