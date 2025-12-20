import os
import sys
import argparse

from rayroom import (
    HybridRenderer,
)
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
    Main function to run the hybrid simulation.
    """
    # 1. Define Small Room (Same as small room example)
    room, sources, mic = DemoRoom(mic_type=mic_type).create_room()
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 4. Save layout visualization
    generate_layouts(room, output_dir, "hybrid")

    # Save mesh if requested
    if save_mesh_flag:
        save_room_mesh(room, output_dir, "hybrid")

    # 5. Setup Hybrid Renderer
    print("Initializing Hybrid Renderer...")
    renderer = HybridRenderer(room, fs=DEFAULT_SAMPLING_RATE, temperature=20.0, humidity=50.0)

    # Assign Audio Files
    print("Assigning audio files...")
    base_path = "audio_sources"

    if not os.path.exists(os.path.join(base_path, "speaker_1.wav")):
        print("Error: Example audio files not found.")
        exit(1)

    renderer.set_source_audio(src1, os.path.join(base_path, "speaker_1.wav"), gain=1.0)
    renderer.set_source_audio(src2, os.path.join(base_path, "speaker_2.wav"), gain=1.0)
    renderer.set_source_audio(src_bg, os.path.join(base_path, "foreground.wav"), gain=0.1)

    # 7. Render using Hybrid Method
    # ism_order=2 means reflections of order 0, 1, 2 are handled by ISM.
    # RayTracer will skip specular reflections <= 2.
    print("Starting Hybrid Rendering pipeline (ISM Order 2 + Ray Tracing)...")

    with PerformanceMonitor() as monitor:
        outputs, _, rirs = renderer.render(
            n_rays=20000,
            max_hops=50,
            rir_duration=1.5,
            record_paths=True,
            interference=False,
            ism_order=2,         # Enable Hybrid Mode
            show_path_plot=False
        )
    save_performance_metrics(monitor, output_dir, "hybrid")

    # 8. Save Result
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is not None:
        run_metrics_and_save(
            mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE,
            output_dir, "hybrid", save_rir_flag=save_rir_flag,
            save_audio_flag=save_audio_flag, save_acoustics_flag=save_acoustics_flag,
            save_psychoacoustics_flag=save_psychoacoustics_flag
        )
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a hybrid simulation with different microphone types."
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
