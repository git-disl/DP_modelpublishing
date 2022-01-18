from __future__ import division

import numpy as np
def exponential_decay(sigma, global_step, decay_rate, decay_steps=1, staircase=False):
    """Applies expoential decay to the noise level"""
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")
    p=global_step/decay_steps
    if staircase:
        p = np.floor(p)
    return sigma*np.exp(-decay_rate*p)

def polynomial_decay(sigma, global_step, decay_steps=1, end_sigma=0.0001, power=1.0):
    """Applies a polynomial decay to the sigma."""
    if global_step is None:
        raise ValueError("global_step is required for polynomial_decay.")
    p = global_step/decay_steps
    return (sigma - end_sigma)*np.power(1-p, power)+end_sigma

def step_decay(sigma, global_step, decay_rate, decay_steps=1):
    if global_step is None:
        raise ValueError("global_step is required for polynomial_decay.")
    return sigma*np.power(decay_rate, np.floor(global_step/decay_steps))

def piecewise_constant(x, boundaries, values):
    """Piecewise constant from boundaries and interval values."""
    default = None
    if x <= boundaries[0]:
        default = values[0]
    elif x > boundaries[-1]:
        default = values[-1]
    for low, high, v in zip(boundaries[:-1], boundaries[1:], values[1:-1]):
        if (x > low) & (x <= high):
            default = v
    return default

def inverse_time_decay(sigma, global_step, decay_rate, decay_steps=1, staircase=False):
    if global_step is None:
        raise ValueError("global_step is required for inverse_time_decay.")
    p = global_step / decay_steps
    if staircase:
      p = np.floor(p)
    denom = 1+decay_rate*p
    return sigma/denom

def consthenexp_decay(sigma, global_step, decay_rate, changestep=None, decay_steps=1, staircase=False):
    """Applies expoential decay to the noise level"""
    if global_step is None:
        raise ValueError("global_step is required for exponential_decay.")

    if changestep == None:
        dif = 0
    else:
        if changestep>global_step:
            return sigma
        else:
            dif = changestep

    p=(global_step-dif)/decay_steps
    if staircase:
        p = np.floor(p)
    return sigma*np.exp(-decay_rate*p)


if __name__=="__main__":
    privacy=0
    for epoch in range(60):
        #sigma =inverse_time_decay(10.0, epoch, 0.0132)
        sigma = step_decay(10.0, epoch, 0.891, decay_steps=10)
        #sigma=exponential_decay(10.0, epoch, 0.014)
        #sigma=polynomial_decay(10.0, epoch, decay_steps=100, end_sigma=2, power=1.432)
        privacy = privacy+ 1.0/(2.0*sigma**2)
        print(sigma, privacy)

