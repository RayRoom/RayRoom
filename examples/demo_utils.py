import os
import json
import numpy as np
from scipy.io import wavfile

from rayroom import (
    Room,
    Source,
    Receiver,
    AmbisonicReceiver,
    get_material,
    Person,
    ThreeSeatCouch,
    CoffeeTable,
    Chair,
)
from rayroom.analytics.acoustics import (
    calculate_clarity,
    calculate_drr,
    calculate_edt,
    calculate_rt60,
    schroeder_integration,
    calculate_loudness,
    calculate_sharpness,
    calculate_roughness,
)
from rayroom.room.visualize import (
    plot_reverberation_time,
    plot_decay_curve,
    plot_spectrogram,
)
from rayroom.effects import presets


DEFAULT_SAMPLING_RATE = 44100


def create_demo_room(mic_type='mono'):
    """
    Creates a standard demo room with furniture, sources, and a receiver.

    Parameters
    ----------
    mic_type : str, optional
        The type of microphone to use, by default 'mono'.
        Can be 'mono' or 'ambisonic'.

    Returns
    -------
    tuple
        A tuple containing:
        - room (Room): The configured room object.
        - sources (dict): A dictionary of sources.
        - mic (Receiver or AmbisonicReceiver): The receiver object.
    """
    # 1. Define Room for Raytracing (8 square meters -> e.g., 4m x 2m or 2.83m x 2.83m)
    # Using 4m x 2m x 2.5m height
    print("Creating room for raytracing demo (4m x 2m x 2.5m)...")
    room = Room.create_shoebox([4, 2, 2.5], materials={
        "floor": get_material("carpet"),
        "ceiling": get_material("plaster"),
        "walls": get_material("concrete")
    }, fs=DEFAULT_SAMPLING_RATE)

    # 2. Add Receiver (Microphone) - centered
    mic_pos = [2, 1, 1.5]
    if mic_type == 'ambisonic':
        print("Using Ambisonic Receiver.")
        mic = AmbisonicReceiver("AmbiMic", mic_pos, radius=0.02)
    else:
        print("Using Mono Receiver.")
        mic = Receiver("MonoMic", mic_pos, radius=0.15)
    room.add_receiver(mic)

    # 4. Add Furniture
    # Mic at (2, 1)
    # Person at (1.2, 1)
    person_1 = Person("Person 1", [0.5, 1.5, 0], rotation_z=90,
                      height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person_1)

    person_2 = Person("Person 2", [3.5, 1.5, 0], rotation_z=-90,
                      height=1.7, width=0.5, depth=0.3, material_name="human")
    room.add_furniture(person_2)

    chair = Chair("Chair", [0.5, 0.65, 0], rotation_z=-45, material_name="wood")
    room.add_furniture(chair)

    couch = ThreeSeatCouch("Couch", [2.9, 0.6, 0], rotation_z=5)
    room.add_furniture(couch)

    coffee_table = CoffeeTable("CoffeeTable", [2.5, 1.5, 0], rotation_z=5)
    room.add_furniture(coffee_table)

    # 5. Define Sources
    # Speaker 1 at one end
    src1 = Source("Speaker 1", [0.5, 1.5, 1.5], power=1.0, orientation=[1, 0, 0], directivity="cardioid")
    # Speaker 2 at the other end
    src2 = Source("Speaker 2", [3.5, 1.5, 1.5], power=1.0, orientation=[-1, 0, 0], directivity="cardioid")
    # Background noise near ceiling
    src_bg = Source("Background Noise", [0.1, 0.1, 2.4], power=0.5)

    room.add_source(src1)
    room.add_source(src2)
    room.add_source(src_bg)

    sources = {
        "src1": src1,
        "src2": src2,
        "src_bg": src_bg
    }

    return room, sources, mic


def save_audio_files(mixed_audio, mic_type, fs, output_dir, filename_prefix):
    """Saves the rendered audio to a WAV file."""
    if mixed_audio is not None:
        output_filename = f"{filename_prefix}_{mic_type}.wav"
        output_path = os.path.join(output_dir, output_filename)
        # Normalize audio
        mixed_audio /= np.max(np.abs(mixed_audio))

        if mic_type == 'ambisonic':
            wavfile.write(output_path, fs, mixed_audio.astype(np.float32))
        else:
            wavfile.write(output_path, fs, (mixed_audio * 32767).astype(np.int16))
        print(f"Simulation complete. Saved to {output_path}")
    else:
        print("Error: No audio output generated.")


def save_rir(rir, mic_type, fs, output_dir, filename_prefix):
    """Saves the RIR to a WAV file."""
    if rir is not None:
        output_filename = f"{filename_prefix}_{mic_type}_rir.wav"
        output_path = os.path.join(output_dir, output_filename)
        # Normalize RIR
        rir = rir / np.max(np.abs(rir))

        if mic_type == 'ambisonic':
            wavfile.write(output_path, fs, rir.astype(np.float32))
        else:
            wavfile.write(output_path, fs, (rir * 32767).astype(np.int16))
        print(f"RIR saved to {output_path}")
    else:
        print("Error: No RIR generated.")


def generate_layouts(room, output_dir, filename_prefix):
    """Generates and saves room layout visualizations."""
    print(f"Saving room layout visualization to {output_dir}...")
    room.plot(os.path.join(output_dir, f"{filename_prefix}_layout.png"), show=False)
    room.plot(os.path.join(output_dir, f"{filename_prefix}_layout_2d.png"), show=False, view='2d')


def save_room_mesh(room, output_dir, filename_prefix):
    """Saves the room geometry as an OBJ mesh file."""
    print(f"Saving room mesh to {output_dir}...")
    mesh_path = os.path.join(output_dir, f"{filename_prefix}_mesh.obj")
    room.save_mesh(mesh_path)
    # Also save the HTML viewer
    viewer_path = os.path.join(output_dir, f"{filename_prefix}_mesh_viewer.html")
    room.save_mesh_viewer(mesh_path, viewer_path)


def compute_and_save_metrics(rir, mixed_audio, mic_name, mic_type, fs, output_dir, filename_prefix):
    """Computes, prints, and saves acoustic metrics and plots."""
    if mic_type == 'ambisonic' and rir.ndim > 1:
        rir = rir[:, 0]

    # 1. RT60 vs. Frequency
    rt_path = os.path.join(output_dir, f"{filename_prefix}_{mic_name}_reverberation_time.png")
    plot_reverberation_time(rir, fs, filename=rt_path, show=False)
    # 2. Decay curve for one octave band (e.g., 1000 Hz)
    decay_path = os.path.join(output_dir, f"{filename_prefix}_{mic_name}_decay_curve_1000hz.png")
    plot_decay_curve(rir, fs, band=1000, schroeder=False, filename=decay_path, show=False)
    # 3. Schroeder curve (broadband)
    schroeder_path = os.path.join(output_dir, f"{filename_prefix}_{mic_name}_schroeder_curve.png")
    plot_decay_curve(rir, fs, schroeder=True, filename=schroeder_path, show=False)

    # Calculate and print metrics
    sch_db = schroeder_integration(rir)
    edt = calculate_edt(sch_db, fs)
    rt60 = calculate_rt60(sch_db, fs)
    c50 = calculate_clarity(rir, fs, 50)
    c80 = calculate_clarity(rir, fs, 80)
    drr = calculate_drr(rir, fs)

    print("\nAcoustic Metrics:")
    print(f"  - EDT (Early Decay Time): {edt:.2f} s")
    print(f"  - RT60 (Reverberation Time): {rt60:.2f} s")
    print(f"  - C50 (Speech Clarity): {c50:.2f} dB")
    print(f"  - C80 (Music Clarity):  {c80:.2f} dB")
    print(f"  - DRR (Direct-to-Reverberant Ratio): {drr:.2f} dB")

    # Save metrics to JSON
    metrics = {
        "edt_s": edt,
        "rt60_s": rt60,
        "c50_db": c50,
        "c80_db": c80,
        "drr_db": drr,
    }
    metrics_path = os.path.join(output_dir, f"{filename_prefix}_{mic_name}_acoustic_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Acoustic metrics saved to {metrics_path}")
    # Plot Spectrogram
    plot_audio = mixed_audio[:, 0] if mic_type == 'ambisonic' else mixed_audio
    plot_spectrogram(
        plot_audio,
        fs,
        title=f"Output Spectrogram ({filename_prefix.capitalize()}, {mic_type})",
        filename=os.path.join(output_dir, f"{filename_prefix}_spectrogram_{mic_type}.png"),
        show=False,
    )


def compute_and_save_psychoacoustic_metrics(mixed_audio, fs, output_dir, filename_prefix, mic_name):
    """Computes, prints, and saves psychoacoustic metrics."""
    if mixed_audio is None:
        return

    print("\nPsychoacoustic Metrics:")

    # Use mono signal for psychoacoustic metrics
    audio_for_metrics = mixed_audio[:, 0] if mixed_audio.ndim > 1 else mixed_audio

    # Loudness
    loudness, _ = calculate_loudness(audio_for_metrics, fs)
    print(f"  - Loudness: {loudness:.2f} sones")

    # Sharpness
    sharpness = calculate_sharpness(audio_for_metrics, fs)
    print(f"  - Sharpness: {sharpness:.2f} acum")

    # Roughness
    roughness_array, _ = calculate_roughness(audio_for_metrics, fs)
    # The new mosqito version may return roughness as an array over time.
    # We take the mean to get a single value for reporting.
    roughness = np.mean(roughness_array) if isinstance(roughness_array, np.ndarray) else roughness_array
    print(f"  - Roughness: {roughness:.2f} asper")

    # Save metrics to JSON
    metrics = {
        "loudness_sones": loudness,
        "sharpness_acum": sharpness,
        "roughness_asper": roughness,
    }
    metrics_path = os.path.join(output_dir, f"{filename_prefix}_{mic_name}_psychoacoustic_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Psychoacoustic metrics saved to {metrics_path}")


def process_effects_and_save(mixed_audio, rir, mic_name, mic_type, fs,
                             output_dir, simulation_name, effects=None,
                             save_rir_flag=False, save_audio_flag=False,
                             save_acoustics_flag=False,
                             save_psychoacoustics_flag=False):
    """
    Processes different audio effects, saves the audio, and computes metrics for each.
    """

    effects_to_process = effects if effects is not None else ["original"]

    for effect in effects_to_process:
        effected_audio = mixed_audio.copy()
        filename_prefix = f"{simulation_name}_simulation_{effect}"

        if effect and effect != "original":
            print(f"Applying effect: {effect}")
            effect_func = presets.get_effect(effect)
            if effect_func:
                effected_audio = effect_func(effected_audio, fs)
            else:
                print(f"Warning: Effect '{effect}' not found.")
                continue

            current_output_dir = os.path.join(output_dir, effect)
        else:
            # "original" case
            current_output_dir = os.path.join(output_dir, "original")

        os.makedirs(current_output_dir, exist_ok=True)

        if save_rir_flag:
            save_rir(rir, mic_name, fs, output_dir, simulation_name)
        if save_audio_flag:
            save_audio_files(effected_audio, mic_type, fs, current_output_dir, filename_prefix)
        if save_acoustics_flag:
            compute_and_save_metrics(rir, effected_audio, mic_name, mic_type, fs, current_output_dir, simulation_name)
        if save_psychoacoustics_flag:
            compute_and_save_psychoacoustic_metrics(effected_audio, fs, current_output_dir, simulation_name, mic_name)


def save_performance_metrics(monitor, output_dir, demo_name):
    """Saves performance metrics to a JSON file.

    Args:
        monitor (PerformanceMonitor): The performance monitor object.
        output_dir (str): The directory to save the metrics to.
        demo_name (str): The name of the demo.
    """
    print("\nPerformance Metrics:")
    print(f"  - Simulation Time: {monitor.runtime_s:.2f}s")
    print(f"  - Peak Memory Usage: {monitor.peak_memory_mb:.2f}MB")

    # Save performance metrics
    perf_metrics = {
        "runtime_s": monitor.runtime_s,
        "peak_memory_mb": monitor.peak_memory_mb,
    }
    perf_metrics_path = os.path.join(output_dir, f"{demo_name}_performance_metrics.json")
    with open(perf_metrics_path, 'w') as f:
        json.dump(perf_metrics, f, indent=4)
    print(f"Performance metrics saved to {perf_metrics_path}\n")
