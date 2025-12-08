"""
This module provides a collection of preset audio effects.

Each function in this module takes an audio signal and a sample rate as input,
and returns the processed audio signal with the effect applied. These presets
are built using the `pedalboard` library and can be used for creative sound
design or to simulate different acoustic environments.

The `EFFECTS` dictionary provides a convenient way to access these presets by name.
"""
import numpy as np
from pedalboard import (
    Pedalboard,
    Compressor,
    Delay,
    Distortion,
    Gain,
    HighpassFilter,
    LowpassFilter,
    Phaser,
    Reverb,
    NoiseGate,
)


def apply_vocal_enhancement_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Applies a vocal enhancement chain to an audio signal.

    This effect preset is designed to improve the clarity and presence of
    vocals in a mix. It includes a noise gate to reduce background noise,
    a high-pass filter to remove low-frequency rumble, a compressor to even
    out dynamics, and a gain stage to boost the overall level.

    Responsibilities:
      * Reduce noise and unwanted low frequencies.
      * Control dynamics for a more consistent vocal level.
      * Increase the perceived loudness of the vocals.

    Example:

        .. code-block:: python

            import numpy as np
            import soundfile as sf
            import rayroom as rt

            # Assume 'vocal.wav' is an audio file with speech or singing
            audio, sample_rate = sf.read('vocal.wav')

            enhanced_audio = rt.effects.presets.apply_vocal_enhancement_effect(
                audio, sample_rate
            )

            sf.write('vocal_enhanced.wav', enhanced_audio, sample_rate)

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        NoiseGate(threshold_db=-30, ratio=1.5, release_ms=250),
        HighpassFilter(cutoff_frequency_hz=80),
        Compressor(threshold_db=-16, ratio=3),
        Gain(gain_db=3)
    ])
    return board(audio, sample_rate)


def apply_telephone_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simulates the sound of a telephone.

    This effect mimics the characteristic sound of a telephone line by
    band-limiting the audio signal and adding a small amount of distortion.

    Responsibilities:
      * Apply high-pass and low-pass filters to create a narrow bandwidth.
      * Add distortion to simulate a lo-fi analog sound.

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal with a telephone effect.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=300),
        LowpassFilter(cutoff_frequency_hz=3400),
        Distortion(drive_db=12),
        Gain(gain_db=-3)
    ])
    return board(audio, sample_rate)


def apply_radio_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simulates the sound of an AM radio broadcast.

    This effect creates the impression of audio being played through an AM
    radio by applying band-pass filtering, compression, and distortion.

    Responsibilities:
      * Limit the frequency range to mimic a radio speaker.
      * Apply heavy compression, characteristic of radio broadcasts.
      * Add distortion for a slightly gritty, analog feel.

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal with a radio effect.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        HighpassFilter(cutoff_frequency_hz=200),
        LowpassFilter(cutoff_frequency_hz=3000),
        Compressor(threshold_db=-15, ratio=5),
        Distortion(drive_db=10),
        Gain(gain_db=-3)
    ])
    return board(audio, sample_rate)


def apply_radio_2_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Simulates a noisy AM radio with a defined noise floor.

    This is a variation of the radio effect that includes more aggressive
    processing, including a noise gate and more pronounced distortion, to
    simulate a lower-quality or more distant radio signal.

    Responsibilities:
      * Gate the signal to manage a simulated noise floor.
      * Apply more extreme filtering and compression.
      * Increase distortion for a more pronounced effect.

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal with a noisy radio effect.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        NoiseGate(threshold_db=-35, ratio=2),
        HighpassFilter(cutoff_frequency_hz=300),
        LowpassFilter(cutoff_frequency_hz=2500),
        Compressor(threshold_db=-24, ratio=5, attack_ms=100, release_ms=1000),
        Distortion(drive_db=15),
        Gain(gain_db=-5)
    ])
    return board(audio, sample_rate)


def apply_large_hall_reverb_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Adds a large hall reverberation effect.

    This preset simulates the acoustics of a large, reverberant space like a
    concert hall or a cathedral. It is useful for adding a sense of space
    and grandeur to an audio signal.

    Responsibilities:
      * Apply a reverb effect with a long decay time.
      * Set parameters to mimic a large room size.

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal with reverb.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        Reverb(room_size=0.9, damping=0.7, wet_level=0.4, dry_level=0.5, width=0.8)
    ])
    return board(audio, sample_rate)


def apply_ambient_wash_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Creates a dreamy, ambient wash effect.

    This effect combines long delays and lush reverb to create a sense of
    ambience and texture. It is well-suited for creating soundscapes or
    adding a dreamy quality to melodic instruments.

    Responsibilities:
      * Use delay with feedback to create repeating echoes.
      * Apply a large reverb to wash out the sound.
      * Filter high frequencies to create a warmer, less defined sound.

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal with an ambient effect.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        Delay(delay_seconds=0.5, feedback=0.4, mix=0.5),
        Reverb(room_size=0.95, damping=0.8, wet_level=0.5, dry_level=0.5),
        LowpassFilter(cutoff_frequency_hz=6000),
    ])
    return board(audio, sample_rate)


def apply_phaser_effect(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Adds a classic swirling phaser effect.

    This effect creates a sweeping, modulated sound that is popular for
    electric guitars, keyboards, and other instruments. It adds movement
    and character to the sound.

    Responsibilities:
      * Apply a phaser effect with a moderate rate and depth.

    :param audio: The input audio signal.
    :type audio: np.ndarray
    :param sample_rate: The sample rate of the audio signal.
    :type sample_rate: int
    :return: The processed audio signal with a phaser effect.
    :rtype: np.ndarray
    """
    board = Pedalboard([
        Phaser(rate_hz=0.5, depth=0.5, mix=0.5)
    ])
    return board(audio, sample_rate)


EFFECTS = {
    "vocal_enhancement": apply_vocal_enhancement_effect,
    "telephone": apply_telephone_effect,
    "radio": apply_radio_effect,
    "radio_2": apply_radio_2_effect,
    "large_hall_reverb": apply_large_hall_reverb_effect,
    "ambient_wash": apply_ambient_wash_effect,
    "phaser": apply_phaser_effect,
}


def get_effect(name: str):
    """Retrieves an effect function by its name.

    This function provides a simple way to access the preset effect functions
    defined in this module. It looks up the requested effect in the `EFFECTS`
    dictionary.

    Example:

        .. code-block:: python

            import rayroom as rt

            # Get the telephone effect function
            telephone_effect = rt.effects.presets.get_effect("telephone")

            # Now, `telephone_effect` can be used to process audio
            # e.g., processed_audio = telephone_effect(audio, sample_rate)

    :param name: The name of the desired effect.
    :type name: str
    :return: The corresponding effect function, or `None` if the name is not found.
    :rtype: callable or None
    """
    return EFFECTS.get(name)
