import os
import sys
import argparse

from rayroom.engines.ism import ImageSourceRenderer
from rayroom.analytics.performance import PerformanceMonitor
from demo_utils import (
    generate_layouts,
    save_room_mesh,
    run_metrics_and_save,
    save_performance_metrics,
)
from rayroom.room.database import DemoRoom
from rayroom.core.constants import DEFAULT_SAMPLING_RATE

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs',
         save_rir_flag=False, save_audio_flag=True, save_acoustics_flag=True,
         save_psychoacoustics_flag=False, save_mesh_flag=False):
    """
    Main function to run the ISM (Image Source Method) only simulation.
    This effectively uses the HybridRenderer with Ray Tracing disabled (n_rays=0).
    """
    # 1. Define Room
    room, sources, mic = DemoRoom(mic_type=mic_type).create_room()
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 4. Save layout visualization
    generate_layouts(room, output_dir, "ism")

    # Save mesh if requested
    if save_mesh_flag:
        save_room_mesh(room, output_dir, "ism")

    # 5. Setup Renderer (Using dedicated ImageSourceRenderer)
    print("Initializing ISM Renderer...")
    renderer = ImageSourceRenderer(room, fs=DEFAULT_SAMPLING_RATE, temperature=20.0, humidity=50.0)

    # Assign Audio Files
    print("Assigning audio files...")
    # build the path to the audio files folder relative to this script file
    base_path = os.path.join(os.path.dirname(__file__), "audio_sources")

    if not os.path.exists(os.path.join(base_path, "speaker_1.wav")):
        print("Error: Example audio files not found.")
        exit(1)

    renderer.set_source_audio(src1, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    renderer.set_source_audio(src2, os.path.join(base_path, "speaker_2.wav"), gain=1.0)
    renderer.set_source_audio(src_bg, os.path.join(base_path, "foreground.wav"), gain=0.1)

    # 7. Render using ISM Method Only
    print("Starting ISM Rendering pipeline...")

    # Note: High ISM orders are computationally expensive (exponential growth).
    # Order 5 or 6 is usually a practical limit for full RIRs in complex rooms,
    # but for simple rooms it can go higher.
    ism_order = 5 
    print(f"  - ISM Order: {ism_order}")

    with PerformanceMonitor() as monitor:
        outputs, rirs = renderer.render(
            ism_order=ism_order,
            rir_duration=1.5,
            interference=False  # Set to True for phase interference effects if desired
        )
    save_performance_metrics(monitor, output_dir, "ism")

    # 8. Save Result
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is not None:
        run_metrics_and_save(
            mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE,
            output_dir, "ism", save_rir_flag=save_rir_flag,
            save_audio_flag=save_audio_flag, save_acoustics_flag=save_acoustics_flag,
            save_psychoacoustics_flag=save_psychoacoustics_flag
        )
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render an ISM-only simulation with different microphone types."
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
        '--save-mesh',
        action='store_true',
        help="Save the room geometry as an OBJ mesh file."
    )
    parser.set_defaults(save_audio=True, save_acoustics=True, save_psychoacoustics=False, save_mesh=False)
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir,
         save_rir_flag=args.save_rir, save_audio_flag=args.save_audio,
         save_acoustics_flag=args.save_acoustics,
         save_psychoacoustics_flag=args.save_psychoacoustics,
         save_mesh_flag=args.save_mesh)

