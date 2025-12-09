import os
import sys
import argparse
import numpy as np

from rayroom import (
    SpectralRenderer,
)
from rayroom.analytics.performance import PerformanceMonitor
from rayroom.effects import presets
from rayroom.room.database import DemoRoom
from demo_utils import (
    generate_layouts,
    save_room_mesh,
    process_effects_and_save,
    save_performance_metrics,
)
from rayroom.core.constants import DEFAULT_SAMPLING_RATE

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs/spectral', effects=None,
         save_rir_flag=False, save_audio_flag=True, save_acoustics_flag=True,
         save_psychoacoustics_flag=False, save_mesh_flag=True):
    """
    Main function to run the spectral simulation.
    """
    # 1. Define Small Room
    room, sources, mic = DemoRoom(mic_type=mic_type).create_room()
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save layout visualization
    generate_layouts(room, output_dir, "spectral")

    # Save mesh if requested
    if save_mesh_flag:
        save_room_mesh(room, output_dir, "spectral")

    # 6. Setup Spectral Renderer
    # Crossover at 500 Hz.
    # FDTD will handle < 500Hz (Diffraction important here).
    # Geometric will handle > 500Hz.
    print("Initializing Spectral Renderer (Crossover: 500 Hz)...")
    renderer = SpectralRenderer(room, fs=DEFAULT_SAMPLING_RATE, crossover_freq=500.0)

    # 7. Assign Audio
    print("Assigning audio...")
    base_path = "audios-trump-indextts15"  # Uncomment if needed
    # base_path = "audios-indextts" # Uncomment if needed
    # base_path = "audios"

    audio_file_1 = os.path.join(base_path, "speaker_1.wav")
    audio_file_2 = os.path.join(base_path, "speaker_2.wav")

    if not os.path.exists(audio_file_1):
        print("Warning: Audio file not found. Creating dummy sine sweep.")
        # Create dummy audio
        t = np.linspace(0, 1, DEFAULT_SAMPLING_RATE)
        audio = np.sin(2 * np.pi * 200 * t * t)  # Sweep
        renderer.set_source_audio(src1, audio, gain=1.0)
        renderer.set_source_audio(src2, audio, gain=1.0)
    else:
        renderer.set_source_audio(src1, audio_file_1, gain=1.0)
        if os.path.exists(audio_file_2):
            renderer.set_source_audio(src2, audio_file_2, gain=1.0)
        else:
            # Use same for src2 if src2 wav missing
            renderer.set_source_audio(src2, audio_file_1, gain=1.0)

    audio_file_bg = os.path.join(base_path, "foreground.wav")
    if os.path.exists(audio_file_bg):
        renderer.set_source_audio(src_bg, audio_file_bg, gain=0.1)

    # 8. Render
    print("Starting Spectral Rendering pipeline...")
    print("Phase 1: HF (Geometric) + Phase 2: LF (FDTD)")
    print("Note: FDTD step may take time...")

    with PerformanceMonitor() as monitor:
        outputs, rirs = renderer.render(
            n_rays=10000,
            max_hops=10,
            rir_duration=1.0,  # Short duration for demo speed
            record_paths=False,
            ism_order=1,
            show_path_plot=False
        )
    save_performance_metrics(monitor, output_dir, "spectral")

    # 9. Save Result
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is not None:
        process_effects_and_save(
            mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE,
            output_dir, "spectral", effects, save_rir_flag=save_rir_flag,
            save_audio_flag=save_audio_flag, save_acoustics_flag=save_acoustics_flag,
            save_psychoacoustics_flag=save_psychoacoustics_flag
        )
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a spectral simulation with different microphone types."
    )
    parser.add_argument(
        '--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
        help="Type of microphone to use ('mono' or 'ambisonic')."
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs',
        help="Output directory for saving files."
    )
    parser.add_argument(
        '--save_rir',
        action='store_true',
        help="Save the Room Impulse Response (RIR) as a WAV file."
    )
    parser.add_argument(
        '--no-save-audio',
        action='store_false',
        dest='save_audio',
        help="Do not save the output audio files."
    )
    parser.add_argument(
        '--no-save-acoustics',
        action='store_false',
        dest='save_acoustics',
        help="Do not compute and save acoustic metrics."
    )
    parser.add_argument(
        '--save-psychoacoustics',
        action='store_true',
        help="Compute and save psychoacoustic metrics."
    )
    parser.add_argument(
        '--no-save-mesh',
        action='store_false',
        dest='save_mesh',
        help="Do not save the room geometry as an OBJ mesh file."
    )
    parser.add_argument(
        '--effects',
        type=str,
        nargs='*',
        default=None,
        choices=list(presets.EFFECTS.keys()) + ["original"],
        help="Apply a post-processing effect to the output audio."
    )
    parser.set_defaults(save_audio=True, save_acoustics=True, save_psychoacoustics=False, save_mesh=True)
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir, effects=args.effects, save_rir_flag=args.save_rir,
         save_audio_flag=args.save_audio, save_acoustics_flag=args.save_acoustics,
         save_psychoacoustics_flag=args.save_psychoacoustics, save_mesh_flag=args.save_mesh)
