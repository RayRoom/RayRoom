"""
This module implements a hybrid acoustic rendering engine.

The `HybridRenderer` class combines two popular methods in geometrical acoustics:
  * **Image Source Method (ISM):** Used for accurately calculating early,
    specular reflections. This method is deterministic and precise for
    low-order reflections in simple geometries.
  * **Ray Tracing:** Used for simulating the late, diffuse reverberant tail
    of the room's response. Ray tracing is a stochastic method that is
    efficient for modeling high-order reflections and diffuse scattering.

By combining these two techniques, the hybrid approach aims to capture the
strengths of both, providing a more perceptually accurate and computationally
efficient simulation than either method alone.
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from ..ism import ImageSourceEngine
from ..raytracer.core import RayTracer
from ...core.utils import generate_rir
from ...room.objects import AmbisonicReceiver


class HybridRenderer:
    """A hybrid acoustic renderer combining ISM and Ray Tracing.

    This class orchestrates the simulation process by first running the
    Image Source Method to generate the early part of the Room Impulse
    Response (RIR), and then using a Ray Tracer to generate the late
    reverberation. The results are combined to form a complete RIR.

    Responsibilities:
      * Manage and configure the ISM and Ray Tracing engines.
      * Handle the assignment of audio signals to sources.
      * Run the simulation pipeline for each source.
      * Combine the early and late reflection histograms.
      * Generate the final RIRs for each receiver.
      * Convolve the RIRs with source audio to produce the final output.

    Example:

        .. code-block:: python

            import rayroom as rt
            import numpy as np

            # Create a room with a source and receiver
            room = rt.room.ShoeBox([10, 8, 4])
            source = room.add_source([5, 4, 2])
            receiver = room.add_receiver([2, 2, 2])
            
            # Initialize the hybrid renderer
            renderer = rt.engines.hybrid.HybridRenderer(room)
            
            # Assign an audio signal to the source
            sample_rate = 44100
            source_audio = np.random.randn(sample_rate * 2)  # 2s of white noise
            renderer.set_source_audio(source, source_audio)
            
            # Run the rendering process
            outputs, rirs = renderer.render(
                n_rays=5000,
                ism_order=2,
                rir_duration=1.0
            )
            
            # `outputs` contains the rendered audio for each receiver
            # `rirs` contains the generated RIRs

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
        """
        Initialize the HybridRenderer.

        :param room: The Room object to render.
        :type room: rayroom.room.Room
        :param fs: Sampling rate in Hz. Defaults to 44100.
        :type fs: int
        :param temperature: Temperature in Celsius. Defaults to 20.0.
        :type temperature: float
        :param humidity: Relative humidity in percent. Defaults to 50.0.
        :type humidity: float
        """
        self.room = room
        self.fs = fs
        self._tracer = RayTracer(room, temperature, humidity)
        self.ism_engine = ImageSourceEngine(room, temperature, humidity)
        self.source_audios = {}
        self.source_gains = {}
        self.last_rirs = {}

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
        if data.dtype == np.int16:
            data = data / 32768.0
        elif data.dtype == np.int32:
            data = data / 2147483648.0
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        if fs != self.fs:
            print(f"Warning: Sample rate mismatch {fs} vs {self.fs}. Playback speed will change.")
        return data

    def render(self, n_rays=10000, max_hops=50, rir_duration=2.0,
               verbose=True, record_paths=False, interference=False,
               ism_order=3, show_path_plot=False):
        """Runs the main hybrid rendering pipeline.

        This method executes the simulation for all sources with assigned
        audio. It combines ISM and Ray Tracing, generates RIRs, and
        convolves them with the source audio to produce the final mixed
        output for each receiver.

        :param n_rays: The number of rays to use for the ray tracing part.
                       Defaults to 10000.
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
                             allowing for interference effects. Defaults to `False`.
        :type interference: bool, optional
        :param ism_order: The maximum reflection order for the Image Source Method.
                          Defaults to 3.
        :type ism_order: int, optional
        :param show_path_plot: (Not implemented in this method) If `True`, a plot
                               of the ray paths would be displayed.
        :type show_path_plot: bool, optional
        :return: A tuple containing a dictionary of receiver outputs (audio) and a
                 dictionary of the last computed RIR for each receiver. If
                 `record_paths` is `True`, the ray paths are also returned.
        :rtype: tuple
        """
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        all_paths = {} if record_paths else None
        self.last_rirs = {}
        valid_sources = [s for s in self.room.sources if s in self.source_audios]

        if not valid_sources:
            print("No sources with assigned audio found.")
            if record_paths:
                return receiver_outputs, all_paths
            return receiver_outputs

        for source in valid_sources:
            if verbose:
                print(f"Simulating Source: {source.name} (ISM Order: {ism_order})")

            source_audio = self.source_audios[source]
            gain = self.source_gains[source]

            # Reset histograms
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    for ch in rx.histograms:
                        rx.histograms[ch] = []
                else:
                    rx.amplitude_histogram = []

            # 1. ISM for early reflections
            self.ism_engine.run(source, max_order=ism_order)

            # 2. Ray Tracing for late reverberation
            paths = self._tracer.run(source, n_rays, max_hops, record_paths=record_paths)
            if record_paths and paths:
                all_paths.update(paths)

            # 3. Combine Histograms & Generate RIRs
            for rx in self.room.receivers:
                # Histograms are now combined on the receiver objects
                if isinstance(rx, AmbisonicReceiver):
                    channel_rirs = []
                    processed_channels = []

                    # Generate RIRs for all configured channels
                    for ch_name in rx.channel_names:
                        hist = rx.histograms[ch_name]
                        rir_ch = generate_rir(
                            hist, self.fs, rir_duration, not interference
                        )
                        channel_rirs.append(rir_ch)
                        processed_ch = fftconvolve(
                            source_audio * gain, rir_ch, mode='full'
                        )
                        processed_channels.append(processed_ch)

                    # Store multi-channel RIR
                    rir = np.stack(channel_rirs, axis=1)

                    # Stack processed audio
                    max_len = max(len(p) for p in processed_channels)
                    padded_channels = [np.pad(p, (0, max_len - len(p))) for p in processed_channels]
                    processed = np.stack(padded_channels, axis=1)

                else:
                    rir = generate_rir(rx.amplitude_histogram, self.fs, rir_duration, not interference)
                    processed = fftconvolve(source_audio * gain, rir, mode='full')
                    # Store as list for consistency if needed, but last_rirs expects array
                    # Processed is 1D array here

                # Store the RIR, last source overwrites.
                self.last_rirs[rx.name] = rir

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad and add
                    current = receiver_outputs[rx.name]
                    is_multichannel = processed.ndim > 1
                    num_channels = processed.shape[1] if is_multichannel else 1

                    if len(processed) > len(current):
                        if is_multichannel:
                            padding = np.zeros(
                                (len(processed) - len(current), num_channels)
                            )
                        else:
                            padding = np.zeros(len(processed) - len(current))
                        current = np.concatenate((current, padding))
                        receiver_outputs[rx.name] = current
                    elif len(current) > len(processed):
                        if is_multichannel:
                            padding = np.zeros(
                                (len(current) - len(processed), num_channels)
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
