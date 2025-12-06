import numpy as np
from scipy.signal import fftconvolve
from .hybrid import HybridRenderer
from .radiosity import RadiositySolver
from .audio import generate_rir

class RadiosityRenderer(HybridRenderer):
    """
    A Hybrid Renderer combining ISM (Early Specular) and Radiosity (Late Diffuse).
    Replaces the stochastic Ray Tracing tail with a smooth Radiosity calculation.
    """
    
    def __init__(self, room, fs=44100, temperature=20.0, humidity=50.0, patch_size=0.5):
        super().__init__(room, fs, temperature, humidity)
        self.radiosity_solver = RadiositySolver(room, patch_size=patch_size)
        
    def render(self, ism_order=2, rir_duration=1.5, verbose=True):
        """
        Render using ISM + Radiosity.
        """
        receiver_outputs = {rx.name: None for rx in self.room.receivers}
        
        valid_sources = [s for s in self.room.sources if s in self.source_audios]
        
        for source in valid_sources:
            if verbose:
                print(f"Radiosity Rendering Source: {source.name}")
                
            # 1. ISM (Early Specular)
            # Clear histograms
            for rx in self.room.receivers:
                rx.energy_histogram = []
                
            if verbose:
                print("  Phase 1: ISM (Early Specular)...")
            
            original_sources = self.room.sources
            self.room.sources = [source]
            self._ism_engine.run(max_order=ism_order, verbose=False)
            self.room.sources = original_sources
            
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
                # ISM is precise. Radiosity is statistical/envelope.
                # We add them.
                # Note: Radiosity already accounted for scattering coeff.
                # ISM handles specular.
                # They are complementary energy paths.
                rx.energy_histogram.extend(diffuse_hist)
                
            # 3. Generate RIR and Convolve
            source_audio = self.source_audios[source]
            gain = self.source_gains.get(source, 1.0)
            
            for rx in self.room.receivers:
                # Use random phase for diffuse tail to sound "reverberant"
                # ISM parts should ideally be fixed phase, but generate_rir is global.
                # For MVP, global random phase is acceptable or we split them.
                
                rir = generate_rir(rx.energy_histogram, fs=self.fs, duration=rir_duration, random_phase=True)
                
                processed = fftconvolve(source_audio * gain, rir, mode='full')
                
                if receiver_outputs[rx.name] is None:
                    receiver_outputs[rx.name] = processed
                else:
                    # Pad and add
                    curr = receiver_outputs[rx.name]
                    if len(processed) > len(curr):
                        curr = np.pad(curr, (0, len(processed)-len(curr)))
                    elif len(curr) > len(processed):
                        processed = np.pad(processed, (0, len(curr)-len(processed)))
                    receiver_outputs[rx.name] = curr + processed
                    
        # Normalize
        for k, v in receiver_outputs.items():
            if v is not None:
                m = np.max(np.abs(v))
                if m > 0:
                    receiver_outputs[k] = v / m
                    
        return receiver_outputs

