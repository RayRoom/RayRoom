"""
This module provides a high-level audio rendering pipeline for the ray tracing engine.

The `RaytracingRenderer` class orchestrates the entire process of simulating
the acoustics of a room for a given set of sources and receivers. It manages:

  - Assigning audio data to sound sources.
  - Running the core `RayTracer` to generate energy histograms for each receiver.
  - Converting these histograms into Room Impulse Responses (RIRs).
  - Convolving the RIRs with the source audio.
  - Mixing the results from all sources to produce the final audio output for
    each receiver.
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from .core import RayTracer
from ...room.objects import AmbisonicReceiver
from ...core.utils import generate_rir


class RaytracingRenderer:
    """Handles the audio rendering pipeline for a Room using ray tracing.

    This class provides a convenient interface to run a full acoustic simulation.
    It takes a `Room` object, manages audio sources, runs the ray tracing
    simulation, generates RIRs, and produces the final convolved and mixed audio
    for each receiver in the room.

    Responsibilities:
      * Manage audio data for multiple sound sources.
      * Control the `RayTracer` engine.
      * Generate RIRs from ray tracing histograms.
      * Perform convolution of RIRs with source audio.
      * Mix the audio from multiple sources for each receiver.

    Example:

        .. code-block:: python

            import rayroom as rt
            import numpy as np

            # Create a room with a source and a receiver
            room = rt.room.ShoeBox([8, 6, 3])
            source = room.add_source([4, 3, 1.5])
            receiver = room.add_receiver([2, 2, 1.5])

            # Initialize the renderer
            renderer = rt.engines.raytracer.RaytracingRenderer(room)

            # Assign an audio signal to the source
            sample_rate = 44100
            source_audio = np.random.randn(sample_rate * 2)  # 2s of white noise
            renderer.set_source_audio(source, source_audio)

            # Run the rendering process
            outputs, rirs = renderer.render(n_rays=10000, rir_duration=1.0)

            # `outputs` contains the rendered audio for the receiver
            # `rirs` contains the generated RIR for the receiver

    :param room: The `Room` object to be simulated.
    :type room: rayroom.room.Room
    :param fs: The master sampling rate for the simulation. Defaults to 44100.
    :type fs: int, optional
    :param temperature: The ambient temperature in Celsius. Defaults to 20.0.
    :type temperature: float, optional
    :param humidity: The relative humidity in percent. Defaults to 50.0.
    :type humidity: float, optional
    """

    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0):
        self._tracer = RayTracer(room, temperature, humidity)
        self.room = room
        self.fs = fs
        self.source_audios = {}  # Map source_obj -> audio_array
        self.source_gains = {}  # Map source_obj -> linear gain
        self.last_rirs = {}  # Attribute to store RIRs

    def set_source_audio(self, source, audio, gain=1.0):
        """Assigns an audio signal to a source.

        The audio can be provided as a file path to a WAV file or as a
        NumPy array.

        :param source: The `Source` object to which the audio will be assigned.
        :type source: rayroom.room.objects.Source
        :param audio: The audio data, either as a path to a WAV file or a NumPy array.
        :type audio: str or np.ndarray
        :param gain: A linear gain factor to apply to the audio. Defaults to 1.0.
        :type gain: float, optional
        """
        if isinstance(audio, str):
            # Load from file
            data = self._load_wav(audio)
        else:
            data = np.array(audio)

        self.source_audios[source] = data
        self.source_gains[source] = gain

    def _load_wav(self, path):
        """Loads and prepares a WAV file for simulation.

        This internal method reads a WAV file, converts it to a mono,
        floating-point signal, and checks for sample rate consistency.

        :param path: The file path to the WAV file.
        :type path: str
        :return: The loaded audio data as a NumPy array.
        :rtype: np.ndarray
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")

        fs, data = wavfile.read(path)

        # Convert to float
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0

        # Mono
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)

        # Resample if needed (Basic check only)
        if fs != self.fs:
            print(
                f"Warning: Sample rate mismatch {fs} vs {self.fs}. "
                "Playback speed will change. Resampling not fully implemented."
            )
            # TODO: Implement resampling

        return data

    def render(self, n_rays=20000, max_hops=50, rir_duration=2.0,
               verbose=True, record_paths=False, interference=False):
        """Runs the full ray tracing and audio rendering pipeline.

        This is the main method to execute a simulation. It performs the following steps:
        1. Traces rays for each source with assigned audio.
        2. Generates an energy/amplitude histogram for each receiver.
        3. Converts these histograms into RIRs.
        4. Convolves the source audio with the corresponding RIRs.
        5. Mixes the resulting audio for each receiver.

        :param n_rays: The number of rays to trace for each source.
                       Defaults to 20000.
        :type n_rays: int, optional
        :param max_hops: The maximum number of reflections for each ray.
                         Defaults to 50.
        :type max_hops: int, optional
        :param rir_duration: The duration of the generated RIRs in seconds.
                             Defaults to 2.0.
        :type rir_duration: float, optional
        :param verbose: If `True`, print progress information. Defaults to `True`.
        :type verbose: bool, optional
        :param record_paths: If `True`, the paths of the rays will be recorded
                             and returned. This can use a lot of memory.
                             Defaults to `False`.
        :type record_paths: bool, optional
        :param interference: If `True`, phase is preserved in the RIR generation,
                             allowing for interference effects. Setting to `False`
                             (the default for `generate_rir`) results in an
                             energy-based RIR. Defaults to `False`.
        :type interference: bool, optional
        :return: A tuple containing a dictionary of receiver outputs (audio) and a
                 dictionary of the last computed RIR for each receiver. If
                 `record_paths` is `True`, the ray paths are also returned.
        :rtype: tuple
        """
        # Initialize outputs for each receiver
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        all_paths = {} if record_paths else None
        self.last_rirs = {}  # Reset RIRs

        # Iterate over sources that have audio assigned
        # Only render sources that are in the room AND have audio
        valid_sources = [
            s for s in self.room.sources
            if s in self.source_audios
        ]

        if not valid_sources:
            print("No sources with assigned audio found in the room.")
            return receiver_outputs

        for source in valid_sources:
            if verbose:
                print(f"Simulating Source: {source.name}")

            # Clear receiver histograms for this source
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram, rx.x_histogram, rx.y_histogram, rx.z_histogram = [], [], [], []
                else:
                    rx.amplitude_histogram = []

            # Run ray tracer for the single source
            paths = self._tracer.run(source, n_rays, max_hops, record_paths=record_paths)

            if record_paths and paths:
                all_paths.update(paths)

            # Generate RIR and convolve for each receiver
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    # Generate 4 RIRs for W, X, Y, Z
                    rir_w = generate_rir(
                        rx.w_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )
                    rir_x = generate_rir(
                        rx.x_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )
                    rir_y = generate_rir(
                        rx.y_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )
                    rir_z = generate_rir(
                        rx.z_histogram, fs=self.fs, duration=rir_duration,
                        random_phase=not interference
                    )

                    # Store multi-channel RIR
                    rir = np.stack([rir_w, rir_x, rir_y, rir_z], axis=1)

                    # Convolve each channel
                    processed_w = fftconvolve(
                        source_audio * gain, rir_w, mode='full'
                    )
                    processed_x = fftconvolve(
                        source_audio * gain, rir_x, mode='full'
                    )
                    processed_y = fftconvolve(
                        source_audio * gain, rir_y, mode='full'
                    )
                    processed_z = fftconvolve(
                        source_audio * gain, rir_z, mode='full'
                    )

                    # Stack into a 4-channel array
                    max_len = max(
                        len(processed_w), len(processed_x),
                        len(processed_y), len(processed_z)
                    )

                    def pad(arr, length):
                        if len(arr) < length:
                            return np.pad(arr, (0, length - len(arr)))
                        return arr

                    processed_w = pad(processed_w, max_len)
                    processed_x = pad(processed_x, max_len)
                    processed_y = pad(processed_y, max_len)
                    processed_z = pad(processed_z, max_len)

                    processed = np.stack(
                        [processed_w, processed_x, processed_y, processed_z],
                        axis=1
                    )
                else:  # Standard Receiver
                    rir = generate_rir(
                        rx.amplitude_histogram, fs=self.fs,
                        duration=rir_duration, random_phase=not interference
                    )
                    processed = fftconvolve(
                        source_audio * gain, rir, mode='full'
                    )

                # Accumulate RIRs - for now, last source overwrites
                self.last_rirs[rx.name] = rir

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Correctly pad and add signals
                    current = receiver_outputs[rx.name]
                    is_ambisonic = len(processed.shape) > 1

                    if len(processed) > len(current):
                        if is_ambisonic:
                            padding = np.zeros(
                                (len(processed) - len(current), 4)
                            )
                        else:
                            padding = np.zeros(len(processed) - len(current))
                        current = np.concatenate((current, padding))
                        receiver_outputs[rx.name] = current
                    elif len(current) > len(processed):
                        if is_ambisonic:
                            padding = np.zeros(
                                (len(current) - len(processed), 4)
                            )
                        else:
                            padding = np.zeros(len(current) - len(processed))
                        processed = np.concatenate((processed, padding))

                    receiver_outputs[rx.name] += processed

        # Normalize final output
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] /= np.max(np.abs(audio))

        if record_paths:
            return receiver_outputs, all_paths, self.last_rirs
        return receiver_outputs, self.last_rirs
