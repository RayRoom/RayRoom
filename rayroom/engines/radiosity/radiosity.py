"""
This module provides a hybrid acoustic rendering engine that combines the
Image Source Method (ISM) with Acoustic Radiosity.

This approach leverages the strengths of both methods:
  * **Image Source Method:** Accurately models early specular reflections,
    which are crucial for localization and clarity.
  * **Acoustic Radiosity:** Efficiently models the late, diffuse
    reverberation, which contributes to the sense of space and envelopment.

The `RadiosityRenderer` class inherits from the standard hybrid renderer and
replaces the stochastic ray tracing component with the deterministic,
energy-based radiosity solver for the late reverberant tail.
"""
import numpy as np
from scipy.signal import fftconvolve

from .core import RadiositySolver
from ...core.utils import generate_rir
from ..hybrid.hybrid import HybridRenderer
from ...room.objects import Receiver, AmbisonicReceiver


class RadiosityRenderer(HybridRenderer):
    """A Hybrid Renderer using ISM for early and Radiosity for late reflections.

    This renderer provides a high-quality simulation by combining the precision
    of the Image Source Method for early specular reflections with the
    efficiency of Acoustic Radiosity for modeling the late diffuse tail.
    It replaces the stochastic ray tracing component of a typical hybrid
    engine with the radiosity solver.

    Responsibilities:
      * Orchestrate the ISM and Radiosity solvers.
      * Run ISM to generate the early reflection histogram.
      * Run the Radiosity solver to generate the late diffuse energy history.
      * Merge the results from both solvers.
      * Generate a complete RIR from the combined energy histogram.
      * Convolve the RIR with source audio to produce the final output.

    Example:

        .. code-block:: python

            import rayroom as rt
            import numpy as np

            # Create a room with a source and receiver
            room = rt.room.ShoeBox([10, 8, 3])
            source = room.add_source([5, 4, 1.5])
            receiver = room.add_receiver([2, 2, 1.5])

            # Initialize the radiosity renderer
            renderer = rt.engines.radiosity.RadiosityRenderer(
                room, patch_size=0.6
            )

            # Assign an audio signal to the source
            sample_rate = 44100
            source_audio = np.random.randn(sample_rate)  # 1s of white noise
            renderer.set_source_audio(source, source_audio)

            # Run the rendering process
            outputs, rirs = renderer.render(ism_order=2, rir_duration=1.2)

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
    :param patch_size: The approximate size of the radiosity patches.
                       Defaults to 0.5.
    :type patch_size: float, optional
    """
    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0, patch_size=0.5):
        super().__init__(room, fs, temperature, humidity)
        self.radiosity_solver = RadiositySolver(room, patch_size=patch_size)
        self.last_rirs = {}

    def render(self, ism_order=2, rir_duration=1.5, verbose=True):
        """Runs the hybrid ISM + Radiosity rendering pipeline.

        This method executes the full simulation, combining the early
        reflections from ISM with the late reverberation from Radiosity
        to produce the final audio output for all receivers.

        :param ism_order: The maximum reflection order for the Image Source Method.
                          Defaults to 2.
        :type ism_order: int, optional
        :param rir_duration: The total duration of the generated RIRs in seconds.
                             Defaults to 1.5.
        :type rir_duration: float, optional
        :param verbose: If `True`, print progress information. Defaults to `True`.
        :type verbose: bool, optional
        :return: A tuple containing a dictionary of receiver outputs (audio) and a
                 dictionary of the last computed RIR for each receiver.
        :rtype: tuple[dict, dict]
        """
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        rirs = {rx.name: None for rx in self.room.receivers}
        self.last_rirs = {}  # Reset RIRs

        valid_sources = [s for s in self.room.sources if s in self.source_audios]
        for source in valid_sources:
            if verbose:
                print(f"Radiosity Rendering Source: {source.name}")
            # 1. ISM (Early Specular)
            # Clear histograms
            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram, rx.x_histogram, rx.y_histogram, rx.z_histogram = [], [], [], []
                elif isinstance(rx, Receiver):
                    rx.energy_histogram = []
            if verbose:
                print("  Phase 1: ISM (Early Specular)...")
            self.ism_engine.run(source, max_order=ism_order, verbose=False)

            # 2. Radiosity (Late Diffuse)
            if verbose:
                print("  Phase 2: Radiosity (Late Diffuse)...")
            # Solve energy flow
            # Time step for radiosity needs to be fine enough for RIR but coarse enough for speed.
            # 10ms (0.01s) is common for energy envelopes.
            # But for audio convolution, we need finer structure?
            # No, we reconstruct noise with this envelope.
            # Let's use 5ms.
            dt_rad = 0.005
            energy_history = self.radiosity_solver.solve(source, duration=rir_duration, time_step=dt_rad)
            # Collect at receivers
            for rx in self.room.receivers:
                # Get diffuse histogram
                diffuse_hist = self.radiosity_solver.collect_at_receiver(rx, energy_history, dt_rad)
                # Merge histograms
                # NOTE: For Ambisonic, the diffuse energy from Radiosity is omnidirectional.
                # It will only be added to the W channel. This is a limitation of combining
                # a non-directional method (Radiosity) with a directional one (Ambisonics).
                # The specular reflections from ISM will still be directional.

                # Convert diffuse energy history to amplitude before adding to histograms
                diffuse_amps = [(t, np.sqrt(e)) for t, e in diffuse_hist if e >= 0]

                if isinstance(rx, AmbisonicReceiver):
                    rx.w_histogram.extend(diffuse_amps)
                else:
                    rx.amplitude_histogram.extend(diffuse_amps)
            # 3. Generate RIR and Convolve
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    # Generate 4 RIRs for W, X, Y, Z
                    rir_w = generate_rir(rx.w_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    rir_x = generate_rir(rx.x_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    rir_y = generate_rir(rx.y_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    rir_z = generate_rir(rx.z_histogram, fs=self.fs, duration=rir_duration, random_phase=True)

                    # Store multi-channel RIR
                    rir = np.stack([rir_w, rir_x, rir_y, rir_z], axis=1)

                    # Convolve each channel
                    processed_w = fftconvolve(source_audio * gain, rir_w, mode='full')
                    processed_x = fftconvolve(source_audio * gain, rir_x, mode='full')
                    processed_y = fftconvolve(source_audio * gain, rir_y, mode='full')
                    processed_z = fftconvolve(source_audio * gain, rir_z, mode='full')

                    # Stack into a 4-channel array
                    max_len = max(len(processed_w), len(processed_x), len(processed_y), len(processed_z))

                    def pad(arr, length):
                        if len(arr) < length:
                            return np.pad(arr, (0, length - len(arr)))
                        return arr

                    processed_w = pad(processed_w, max_len)
                    processed_x = pad(processed_x, max_len)
                    processed_y = pad(processed_y, max_len)
                    processed_z = pad(processed_z, max_len)
                    processed = np.stack([processed_w, processed_x, processed_y, processed_z], axis=1)

                else:  # Standard Receiver
                    rir = generate_rir(rx.amplitude_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    processed = fftconvolve(source_audio * gain, rir, mode='full')

                if rirs[rx.name] is None:
                    rirs[rx.name] = rir
                else:
                    # This part is tricky for multiple sources.
                    # RIRs are source-dependent. For now, we overwrite (last source dominates).
                    # A better approach would be to return RIRs per source.
                    rirs[rx.name] = rir

                self.last_rirs[rx.name] = rir

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad and add
                    curr = receiver_outputs[rx.name]
                    is_ambisonic = len(processed.shape) > 1

                    if len(processed) > len(curr):
                        if is_ambisonic:
                            curr = np.pad(curr, ((0, len(processed) - len(curr)), (0, 0)))
                        else:
                            curr = np.pad(curr, (0, len(processed) - len(curr)))
                    elif len(curr) > len(processed):
                        if is_ambisonic:
                            processed = np.pad(processed, ((0, len(curr) - len(processed)), (0, 0)))
                        else:
                            processed = np.pad(processed, (0, len(curr) - len(processed)))

                    receiver_outputs[rx.name] = curr + processed
        # Normalize
        for k, v in receiver_outputs.items():
            if v is not None:
                m = np.max(np.abs(v))
                if m > 0:
                    receiver_outputs[k] = v / m
        return receiver_outputs, self.last_rirs
