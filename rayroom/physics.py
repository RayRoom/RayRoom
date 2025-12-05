import numpy as np

def air_absorption_coefficient(freq, temperature=20.0, humidity=50.0, pressure=101325.0):
    """
    Calculate air absorption coefficient (alpha) in dB/m.
    Simplified approximation proportional to frequency squared.
    Good enough for visualization and basic energy decay.
    
    Args:
        freq (float): Frequency in Hz.
        temperature (float): Temperature in Celsius (unused in simplified model).
        humidity (float): Relative humidity in percent (unused in simplified model).
        pressure (float): Atmospheric pressure in Pascals (unused).
        
    Returns:
        float: Absorption coefficient in dB/m. 
    """
    # Approx 0.005 dB/m at 1kHz
    # Scales with f^2
    alpha_approx = 5e-9 * (freq**2)
    return alpha_approx
