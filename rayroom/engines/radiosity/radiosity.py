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
                    for ch in rx.histograms:
                        rx.histograms[ch] = []
                elif isinstance(rx, Receiver):
                    rx.amplitude_histogram = [] # Note: corrected from energy_histogram to amplitude_histogram to match other engines if needed, but keeping original variable name if it was correct in this context. 
                    # Checking original code: used rx.energy_histogram but ISM/Raytracer use amplitude_histogram? 
                    # Wait, ism.py records to amplitude_histogram. Radiosity might be using a different one or it's a bug in previous code?
                    # Let's check objects.py again. Receiver has amplitude_histogram.
                    # Original code had: rx.energy_histogram = [] at line 119, but later at 151: rx.amplitude_histogram.extend(diffuse_amps). 
                    # So line 119 was likely creating a new attribute that wasn't used later or was a mistake. 
                    # However, ISM records to amplitude_histogram. So we should clear amplitude_histogram.
                    rx.amplitude_histogram = []
            
            if verbose:
                print("  Phase 1: ISM (Early Specular)...")
            self.ism_engine.run(source, max_order=ism_order, verbose=False)

            # 2. Radiosity (Late Diffuse)
            if verbose:
                print("  Phase 2: Radiosity (Late Diffuse)...")
            # Solve energy flow
            # Time step for radiosity needs to be fine enough for RIR but coarse enough for speed.
            dt_rad = 0.005
            energy_history = self.radiosity_solver.solve(source, duration=rir_duration, time_step=dt_rad)
            # Collect at receivers
            for rx in self.room.receivers:
                # Get diffuse histogram
                diffuse_hist = self.radiosity_solver.collect_at_receiver(rx, energy_history, dt_rad)
                # Merge histograms
                # NOTE: For Ambisonic, the diffuse energy from Radiosity is omnidirectional.
                
                # Convert diffuse energy history to amplitude before adding to histograms
                diffuse_amps = [(t, np.sqrt(e)) for t, e in diffuse_hist if e >= 0]

                if isinstance(rx, AmbisonicReceiver):
                    if rx.config in ["1st_order", "2nd_order"]:
                        rx.histograms['W'].extend(diffuse_amps)
                    elif rx.config == "binaural":
                        rx.histograms['L'].extend(diffuse_amps)
                        rx.histograms['R'].extend(diffuse_amps)
                else:
                    rx.amplitude_histogram.extend(diffuse_amps)

            # 3. Generate RIR and Convolve
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)

            for rx in self.room.receivers:
                if isinstance(rx, AmbisonicReceiver):
                    channel_rirs = []
                    processed_channels = []

                    for ch_name in rx.channel_names:
                        hist = rx.histograms[ch_name]
                        rir_ch = generate_rir(
                            hist, fs=self.fs, duration=rir_duration,
                            random_phase=True
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

                    def pad(arr, length):
                        if len(arr) < length:
                            return np.pad(arr, (0, length - len(arr)))
                        return arr
                    
                    padded_channels = [pad(p, max_len) for p in processed_channels]
                    processed = np.stack(padded_channels, axis=1)

                else:  # Standard Receiver
                    rir = generate_rir(rx.amplitude_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                    processed = fftconvolve(source_audio * gain, rir, mode='full')

                if rirs[rx.name] is None:
                    rirs[rx.name] = rir
                else:
                    # RIRs are source-dependent. For now, we overwrite (last source dominates).
                    rirs[rx.name] = rir

                self.last_rirs[rx.name] = rir

                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad and add
                    curr = receiver_outputs[rx.name]
                    is_multichannel = processed.ndim > 1
                    num_channels = processed.shape[1] if is_multichannel else 1

                    if len(processed) > len(curr):
                        if is_multichannel:
                            curr = np.pad(curr, ((0, len(processed) - len(curr)), (0, 0)))
                        else:
                            curr = np.pad(curr, (0, len(processed) - len(curr)))
                    elif len(curr) > len(processed):
                        if is_multichannel:
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
