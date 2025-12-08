import numpy as np


def air_absorption_coefficient(freq, temperature=20.0, humidity=50.0, pressure=101325.0):
    """Calculates the air absorption coefficient based on ISO 9613-1.

    This function determines the attenuation of sound in air due to absorption,
    which is dependent on frequency, temperature, humidity, and atmospheric
    pressure. The result is given in decibels per meter (dB/m).

    Responsibilities:
      * Implement the ISO 9613-1 standard for calculating air absorption.
      * Account for environmental factors like temperature, humidity, and pressure.
      * Calculate relaxation frequencies for oxygen and nitrogen.

    Example:

        .. code-block:: python

            import rayroom as rt

            # Absorption at 1000 Hz in standard conditions
            alpha_1k = rt.core.physics.air_absorption_coefficient(1000)
            print(f"Absorption at 1 kHz: {alpha_1k:.4f} dB/m")
            
            # Absorption in a warmer, more humid environment
            alpha_warm_humid = rt.core.physics.air_absorption_coefficient(
                1000, temperature=30.0, humidity=70.0
            )
            print(f"Absorption at 1 kHz (warm, humid): {alpha_warm_humid:.4f} dB/m")

    :param freq: Frequency in Hertz (Hz).
    :type freq: float
    :param temperature: Ambient temperature in Celsius. Defaults to 20.0.
    :type temperature: float, optional
    :param humidity: Relative humidity in percent (e.g., 50.0 for 50%).
                     Defaults to 50.0.
    :type humidity: float, optional
    :param pressure: Atmospheric pressure in Pascals (Pa). Defaults to 101325.0.
    :type pressure: float, optional
    :return: The absorption coefficient in decibels per meter (dB/m).
    :rtype: float
    """
    # Constants for ISO 9613-1
    p_ref = 101325.0  # Reference pressure (Pa)
    T_ref = 293.15    # Reference temperature (K) (20 C)
    T_triple = 273.16  # Triple point isotherm temperature (K)

    # Convert temperature to Kelvin
    T = temperature + 273.15

    # Ratios
    p_ratio = pressure / p_ref
    T_ratio = T / T_ref

    # Saturation vapor pressure (psat) / p_ref
    # Formula: 10 ^ ( -6.8346 * (T_triple/T)^1.261 + 4.6151 )
    exponent = -6.8346 * ((T_triple / T)**1.261) + 4.6151
    psat_ratio = 10**exponent

    # Molar concentration of water vapor (h) in percent
    h = humidity * psat_ratio / p_ratio

    # Oxygen relaxation frequency (frO)
    frO = p_ratio * (24 + 4.04e4 * h * (0.02 + h) / (0.391 + h))

    # Nitrogen relaxation frequency (frN)
    frN = p_ratio * (T_ratio**-0.5) * (9 + 280 * h * np.exp(-4.170 * ((T_ratio**(-1/3)) - 1)))

    # Attenuation coefficient alpha (dB/m)
    term1 = 1.84e-11 * (p_ratio**-1) * (T_ratio**0.5)
    term2 = (T_ratio**-2.5) * (
        (0.01275 * np.exp(-2239.1 / T) / (frO + (freq**2 / frO))) +
        (0.1068 * np.exp(-3352.0 / T) / (frN + (freq**2 / frN)))
    )

    alpha = 8.686 * (freq**2) * (term1 + term2)

    return alpha
