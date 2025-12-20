"""
This module implements the ImageSourceRenderer, a high-level renderer that
orchestrates the full acoustic simulation pipeline using purely the Image Source Method.
It manages audio assignment, runs the ISM engine, generates RIRs, and convolves
them with source audio.
"""
import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import fftconvolve

from ...core.utils import generate_rir
from ...room.objects import AmbisonicReceiver
from .ism import ImageSourceEngine


class ImageSourceRenderer:
    """A renderer for acoustic simulation using only the Image Source Method (ISM).

    This class provides a high-level interface similar to `HybridRenderer` but
    exclusively uses ISM for all calculations. It manages the simulation process:
    assigning audio, running the ISM engine, generating RIRs from the calculated
    reflections, and convolving these RIRs with the source audio.

    Responsibilities:
      * Manage the ISM engine.
      * Handle audio signal assignment to sources.
      * Run the simulation pipeline.
      * Generate RIRs from ISM histograms.
      * Convolve RIRs with source audio to produce final output.

    :param room: The `Room` object to be simulated.
    :type room: rayroom.room.Room
    :param fs: The sampling rate for the simulation. Defaults to 44100.
    :type fs: int, optional
    :param temperature: The ambient temperature in Celsius. Defaults to 20.0.
    :type temperature: float, optional
    :param humidity: The relative humidity in percent. Defaults to 50.0.
    :type humidity: float, optional
    """

    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0):
        self.room = room
        self.fs = fs
        self.ism_engine = ImageSourceEngine(room, temperature, humidity)
        self.source_audios = {}
        self.source_gains = {}
        self.last_rirs = {}

    def set_source_audio(self, source, audio, gain=1.0):
        """Assigns an audio signal to a source.

        :param source: The `Source` object.
        :type source: rayroom.room.objects.Source
        :param audio: Path to WAV file or NumPy array.
        :type audio: str or np.ndarray
        :param gain: Linear gain factor. Defaults to 1.0.
        :type gain: float, optional
        """
        if isinstance(audio, str):
            data = self._load_wav(audio)
        else:
            data = np.array(audio)
        self.source_audios[source] = data
        self.source_gains[source] = gain

    def _load_wav(self, path):
        """Loads and prepares a WAV file."""
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

    def render(self, ism_order=3, rir_duration=1.0, verbose=True, interference=False):
        """Runs the ISM rendering pipeline.

        :param ism_order: The maximum reflection order for ISM. Defaults to 3.
        :type ism_order: int, optional
        :param rir_duration: Duration of the generated RIRs in seconds. Defaults to 1.0.
        :type rir_duration: float, optional
        :param verbose: If `True`, print progress. Defaults to `True`.
        :type verbose: bool, optional
        :param interference: If `True`, phase is preserved (not fully implemented in RIR gen yet).
                             Defaults to `False`.
        :type interference: bool, optional
        :return: A tuple (receiver_outputs, last_rirs).
        :rtype: tuple
        """
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        self.last_rirs = {}
        valid_sources = [s for s in self.room.sources if s in self.source_audios]

        if not valid_sources:
            print("No sources with assigned audio found.")
            return receiver_outputs, self.last_rirs

        for source in valid_sources:
            if verbose:
                print(f"Simulating Source: {source.name} (ISM Order: {ism_order})")

            # Reset histograms
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram, rx.x_histogram, rx.y_histogram, rx.z_histogram = [], [], [], []
                else:
                    rx.amplitude_histogram = []

            # Run ISM Engine
            self.ism_engine.run(source, max_order=ism_order, verbose=verbose)

            # Generate RIRs & Convolve
            for rx in self.room.receivers:
                # We need to construct RIRs *after* all sources? No, we do it per source to convolve?
                # Actually standard Hybrid does it per source.
                # BUT histograms accumulate if we don't clear them?
                # Ah, HybridRenderer CLEARS histograms for each source loop:
                # "for rx in self.room.receivers: ... rx.amplitude_histogram = []"
                # And here we do the same above. So histograms contain ONLY current source reflections.
                
                if isinstance(rx, AmbisonicReceiver):
                    rir_w = generate_rir(rx.w_histogram, self.fs, rir_duration, not interference)
                    rir_x = generate_rir(rx.x_histogram, self.fs, rir_duration, not interference)
                    rir_y = generate_rir(rx.y_histogram, self.fs, rir_duration, not interference)
                    rir_z = generate_rir(rx.z_histogram, self.fs, rir_duration, not interference)
                    rirs = [rir_w, rir_x, rir_y, rir_z]
                    rir = np.stack(rirs, axis=1)
                else:
                    rir = generate_rir(rx.amplitude_histogram, self.fs, rir_duration, not interference)
                    rirs = [rir]

                self.last_rirs[rx.name] = rir

                # Convolve
                source_audio = self.source_audios[source]
                gain = self.source_gains.get(source, 1.0)

                if isinstance(rx, AmbisonicReceiver):
                    processed_channels = [fftconvolve(source_audio * gain, rir_ch, mode='full') for rir_ch in rirs]
                    max_len = max(len(pc) for pc in processed_channels)
                    padded_channels = [np.pad(pc, (0, max_len - len(pc))) for pc in processed_channels]
                    processed = np.stack(padded_channels, axis=1)
                else:
                    processed = fftconvolve(source_audio * gain, rirs[0], mode='full')

                # Mix into final buffer
                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    current_len = receiver_outputs[rx.name].shape[0]
                    new_len = processed.shape[0]
                    if new_len > current_len:
                        padding_shape = (new_len - current_len,) + receiver_outputs[rx.name].shape[1:]
                        receiver_outputs[rx.name] = np.concatenate([receiver_outputs[rx.name], np.zeros(padding_shape)])
                    elif current_len > new_len:
                        padding_shape = (current_len - new_len,) + processed.shape[1:]
                        processed = np.concatenate([processed, np.zeros(padding_shape)])
                    receiver_outputs[rx.name] += processed

        # Normalize final mix
        for name, audio in receiver_outputs.items():
            if audio is not None and np.max(np.abs(audio)) > 0:
                receiver_outputs[name] /= np.max(np.abs(audio))

        return receiver_outputs, self.last_rirs

