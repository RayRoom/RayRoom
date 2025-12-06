import sys
import os
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from rayroom import Room, Source, Receiver, RadiosityRenderer, get_material

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    FS = 44100

    # 1. Define Room
    print("Creating room (4m x 3m x 2.5m)...")
    room = Room.create_shoebox([4, 3, 2.5], materials={
        "floor": get_material("wood"),
        "ceiling": get_material("plaster"),
        "walls": get_material("brick")
    })

    mic = Receiver("Microphone", [2, 1.5, 1.5], radius=0.15)
    room.add_receiver(mic)

    # Add Speaker 1
    src1 = Source("Speaker 1", [1, 1.5, 1.5], power=1.0)
    room.add_source(src1)

    # Add Speaker 2 (as in render_hybrid.py)
    src2 = Source("Speaker 2", [3.0, 1.5, 1.5], power=1.0)
    room.add_source(src2)

    # 2. Setup Radiosity Renderer
    # Patch size determines grid resolution. 0.5m is coarse but fast.
    print("Initializing Radiosity Renderer...")
    renderer = RadiosityRenderer(room, fs=FS, patch_size=0.5)

    # Assign Audio
    base_path = "examples/audios-trump-indextts15"
    # base_path = "examples/audios-indextts"
    # base_path = "examples/audios"

    audio_file_1 = os.path.join(base_path, "speaker_1.wav")
    audio_file_2 = os.path.join(base_path, "speaker_2.wav")

    if not os.path.exists(audio_file_1):
        print("Warning: Audio file not found. Using sweep.")
        t = np.linspace(0, 1, FS)
        audio = np.sin(2 * np.pi * 200 * t * t)
        renderer.set_source_audio(src1, audio, gain=1.0)
        renderer.set_source_audio(src2, audio, gain=1.0)
    else:
        renderer.set_source_audio(src1, audio_file_1, gain=1.0)
        if os.path.exists(audio_file_2):
            renderer.set_source_audio(src2, audio_file_2, gain=1.0)
        else:
            renderer.set_source_audio(src2, audio_file_1, gain=1.0)

    # 3. Render
    print("Starting Radiosity Rendering (ISM + Energy Exchange)...")

    # ISM Order 2 for early reflections
    outputs = renderer.render(ism_order=2, rir_duration=1.0)

    # 4. Save Result
    mixed_audio = outputs["Microphone"]
    if mixed_audio is not None:
        output_file = "radiosity_simulation.wav"
        # Normalize
        mixed_audio /= np.max(np.abs(mixed_audio))
        wavfile.write(output_file, FS, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {output_file}")

        plt.figure(figsize=(10, 4))
        plt.specgram(mixed_audio, Fs=FS, NFFT=1024, noverlap=512)
        plt.title("Output Spectrogram (Radiosity)")
        plt.colorbar(format='%+2.0f dB')
        plt.savefig("radiosity_spectrogram.png")
        print("Saved spectrogram.")


if __name__ == "__main__":
    main()
