#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for estimating the voltage supply in the piezoelectric patch
"""

import numpy as np

def piezo_voltage_distributed(moment_dist, x, n_p, b_p, E_p, d31, h_s, h_p, E_s=None, eta=1):
    """
    Compute the required voltage for a piezoelectric patch (or patch pair)
    when the bending moment varies along the patch length.

    Parameters:
    -----------
    moment_dist : array_like
        Desired distributed bending moment along patch [N·m/m].
    x : array_like
        Spatial positions along the patch [m]. Same length as moment_dist.
    n_p : int
        Number of active patches (1 = single, 2 = symmetric pair)
    b_p : float
        Patch width [m]
    E_p : float
        Young's modulus of piezo patch [Pa]
    d31 : float
        Piezoelectric strain coefficient [m/V]
    h_s : float
        Beam substrate thickness [m]
    h_p : float
        Patch thickness [m]
    E_s : float, optional
        Young's modulus of substrate [Pa]. If None, assume E_s = E_p.

    Returns:
    --------
    V : float
        Equivalent required voltage [V] to generate the same total moment.
    Theta : float
        Voltage-to-moment gain [N·m/V]
    M_total : float
        Integrated desired bending moment [N·m]
    """
    if E_s is None:
        E_s = E_p

    # Areas of substrate and patch
    A_s = b_p * h_s
    A_p = b_p * h_p

    # Centroid locations (from bottom surface)
    z_s = h_s / 2
    z_t = h_s + h_p / 2
    z_b = h_p / 2

    # Neutral axis location
    z0 = (E_s * A_s * z_s + E_p * A_p * z_t + E_p * A_p * z_b) / (E_s * A_s + 2 * E_p * A_p)

    # Effective lever arm (distance from NA to patch centroid)
    z_eff = abs(z_t - z0)

    # Voltage-to-moment gain per unit length (Theta_tilde)
    Theta_tilde = eta * n_p * b_p * E_p * d31 * z_eff  # [N·m/m/V]

    # Integrate distributed moment to get total desired resultant
    M_total = np.trapz(moment_dist, x)

    # Equivalent required voltage
    V = M_total / (Theta_tilde * (x[-1] - x[0]))

    return V, Theta_tilde, M_total


def max_moment_distributed(voltage_limit, L_p, n_p, b_p, E_p, d31, h_s, h_p, E_s=None):
    """
    Compute maximum achievable resultant bending moment over a patch length,
    given a voltage limit.

    Parameters:
    -----------
    voltage_limit : float
        Maximum safe applied voltage [V]
    L_p : float
        Patch length [m]
    (other parameters: same as piezo_voltage_distributed)

    Returns:
    --------
    M_max : float
        Maximum resultant bending moment [N·m]
    """
    _, Theta_tilde, _ = piezo_voltage_distributed(moment_dist=[1, 1], x=[0, L_p],
                                                  n_p=n_p, b_p=b_p, E_p=E_p,
                                                  d31=d31, h_s=h_s, h_p=h_p, E_s=E_s)
    M_max = Theta_tilde * L_p * voltage_limit
    return M_max


# Example usage:
if __name__ == "__main__":
    # Suppose desired moment distribution is linear along 5 cm patch
    x = np.linspace(0, 0.05, 100)
    m_dist = 10 * x  # N·m/m varying linearly

    V, Theta_tilde, M_total = piezo_voltage_distributed(
        moment_dist=m_dist, x=x, n_p=2, b_p=0.01, E_p=6.3e10,
        d31=-175e-12, h_s=0.001, h_p=0.0003)

    print(f"Total desired moment = {M_total:.4f} N·m")
    print(f"Equivalent required voltage = {V:.2f} V")

    V_limit = 200
    M_max = max_moment_distributed(voltage_limit=V_limit, L_p=0.05, n_p=2, b_p=0.01,
                                   E_p=6.3e10, d31=-175e-12, h_s=0.001, h_p=0.0003)
    print(f"Maximum safe bending moment at {V_limit} V = {M_max:.4f} N·m")
