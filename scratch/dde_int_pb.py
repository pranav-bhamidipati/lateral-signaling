import numpy as np
import scipy.integrate
import scipy.interpolate
import math


def ddeint_pb(func, y0, t, taus, args=(), y0_args=(), n_time_points_per_step=200):
    """Integrate a system of delay differential equations defined by
        y' = f(t, y, y(t-tau1), y(t-tau2), ...)
    using the method of steps.

    Parameters
    ----------
    func : function, call signature func(y, t, y_past, *args)
        Function specifying the right hand side of the system of DDEs.
        Assuming we have a system of `n` DDEs, its arguments are:
        y : Numpy array, shape (n, )
            The current value of y.
        t : float
            The current time.
        y_past : function, call signature y_past(t)
            Function used to compute values of y in the past. This is
            not specified by the user, but called as, e.g., 
            `y_past(t-tau)` within `func`. The function is automatically
            generated using interpolants.
        args : tuple
            Tuple of any other arguments to be passed `func`. Note that
            the values of the delays, `tau`, are usually included in
            `args`.
    y0 : function, call signature y0(t, *y0_args)
        A function to compute the pre- time = t[0] values of `y`.
    t : array_like
        The time points for which the solution of the DDEs is to be 
        returned.
    tau : float
        The longest of all the time delays. This defines the step length
        for the method of steps.
    args : tuple, default ()
        Tuple of arguments to be passed to `func`.
    y0_args : tuple, default ()
        Tuple of arguments to be passed to `y0`.
    n_time_points_per_step : int, default 200
        The number of time points to store the solution for each step.
        These points are then used to compute an interpolant.

    Returns
    -------
    y : array, shape (len(t), len(y0))
    Array containing the value of y for each desired time in t.

    Notes
    -----
    .. Uses `scipy.integrate.odeint()` to integrate each step. To 
       compute the values of `y` at time points from a previous step,
       uses a cubic B-spline interpolant of the solution from the 
       previous step.
    .. `n_time_points_per_step` may be adjusted downward if the value
       of `y` does not change rapidly for a given step, but should be
       large if it does to to able to capture the dynamics.

    """
    
    t0 = t[0]
    nt = t.size
    dt = t[1] - t[0]

    # Get dimension of problem for convenience
    n = np.atleast_2d(y0).shape[1]

    ######
    
    # Get each delay as discrete # steps (step size dt)
    # Sample past function at time-points linspace(-max_tau, 0, dt)
    # Either
    #   - Perform first integration once for each delay, from 
    #       shortest to longest. Issue here is you can still run into
    #       trouble interpolating (e.g. you can get negative values
    #       for nonnegative variables).
    #   - Store the past y0 values at the end of the y_return array
    #       so that they can be accessed using negative indices. 
    #       They will be overwritten in later steps. Add an assert
    #       statement checking that the max # delay steps is less 
    #       than nt, otherwise it would break the strategy.
    #   - Use if-then statement to pull from past y0 if the delay is
    #       referring to the past. Slower than above.
    
    # Main downside of this approach is that I don't think you can 
    #   use Scipy's built-in odeint(). Maybe scipy.integrate.RK45?
    
    ######
    
    # Approx. number of steps for each delay param
    step_taus = np.array([math.ceil(tau / dt) for tau in taus], dtype=int)
    
    assert (np.all(step_taus >= min_delay)), (
        "One or more delays is too short. Lower dt or lengthen delay."
    )
    assert (np.all(step_taus < nt)), (
        "One or more delays is longer than time-span."
    )
    
    # Initialize output
    y = np.empty((nt, n))
    
    # Fill last elements with samples of past time-points
    t_past = np.arange(t[0] - dt * max(step_taus), t[0], dt)
    y[-max(step_taus):] = y0(t_past)
  
    for t_step in range(nt):
        
        y_past = y[t_step - step_taus]
        
        
#     # Integrate subsequent steps
#     j = 1
#     while t_step[-1] < t[-1]:
        # Make B-spline
        tck = [scipy.interpolate.splrep(t_step, y[:, i]) for i in range(n)]

        # Interpolant of y from previous step
        y_past = lambda t: np.array(
            [scipy.interpolate.splev(t, tck[i]) for i in range(n)]
        )

        # Integrate this step
        t_step = np.linspace(t0 + j * tau, t0 + (j + 1) * tau, n_time_points_per_step)
        y = scipy.integrate.odeint(func, y[-1, :], t_step, args=(y_past,) + args)

        # Store the result
        y_dense.append(y[:-1, :])
        t_dense.append(t_step[:-1])

        j += 1

    # Concatenate results from steps
    y_dense = np.concatenate(y_dense)
    t_dense = np.concatenate(t_dense)

    # Interpolate solution for returning
    y_return = np.empty((len(t), n))
    for i in range(n):
        tck = scipy.interpolate.splrep(t_dense, y_dense[:, i])
        y_return[:, i] = scipy.interpolate.splev(t, tck)

    return y_return
    
    
    
    
    
    
    
#     t0 = t[0]
#     y_dense = []
#     t_dense = []

#     # Past function for the first step
#     y_past = lambda t: y0(t, *y0_args)

#     # Integrate first step
#     t_step = np.linspace(t0, t0 + tau, n_time_points_per_step)
#     y = scipy.integrate.odeint(func, y_past(t0), t_step, args=(y_past,) + args)

#     # Store result from integration
#     y_dense.append(y[:-1, :])
#     t_dense.append(t_step[:-1])

#     # Get dimension of problem for convenience
#     n = y.shape[1]

#     # Integrate subsequent steps
#     j = 1
#     while t_step[-1] < t[-1]:
#         # Make B-spline
#         tck = [scipy.interpolate.splrep(t_step, y[:, i]) for i in range(n)]

#         # Interpolant of y from previous step
#         y_past = lambda t: np.array(
#             [scipy.interpolate.splev(t, tck[i]) for i in range(n)]
#         )

#         # Integrate this step
#         t_step = np.linspace(t0 + j * tau, t0 + (j + 1) * tau, n_time_points_per_step)
#         y = scipy.integrate.odeint(func, y[-1, :], t_step, args=(y_past,) + args)

#         # Store the result
#         y_dense.append(y[:-1, :])
#         t_dense.append(t_step[:-1])

#         j += 1

#     # Concatenate results from steps
#     y_dense = np.concatenate(y_dense)
#     t_dense = np.concatenate(t_dense)

#     # Interpolate solution for returning
#     y_return = np.empty((len(t), n))
#     for i in range(n):
#         tck = scipy.interpolate.splrep(t_dense, y_dense[:, i])
#         y_return[:, i] = scipy.interpolate.splev(t, tck)

#     return y_return