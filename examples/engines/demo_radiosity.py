import os
import sys
import argparse

from rayroom import RadiosityRenderer
from rayroom.analytics.performance import PerformanceMonitor
from rayroom.room.database import DemoRoom
from demo_utils import (
    generate_layouts,
    save_room_mesh,
    run_metrics_and_save,
    save_performance_metrics,
)
from rayroom.core.constants import DEFAULT_SAMPLING_RATE

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main(mic_type='mono', output_dir='outputs',
         save_rir_flag=False, save_audio_flag=True, save_acoustics_flag=True,
         save_psychoacoustics_flag=False, save_mesh_flag=True):
    # 1. Define Room
    # room, sources, mic = TestBenchRoom(mic_type=mic_type).create_room()
    room, sources, mic = DemoRoom(mic_type=mic_type).create_room()
    # room, sources, mic = MedicalRoom8M(mic_type=mic_type).create_room()
    src1 = sources["src1"]
    src2 = sources["src2"]
    src_bg = sources["src_bg"]

    # 3. Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Save layout visualization
    generate_layouts(room, output_dir, "radiosity")

    # Save mesh if requested
    if save_mesh_flag:
        save_room_mesh(room, output_dir, "radiosity")

    # 6. Setup Radiosity Renderer
    print("Initializing Radiosity Renderer...")
    renderer = RadiosityRenderer(room, fs=DEFAULT_SAMPLING_RATE)

    # 7. Assign Audio Files
    print("Assigning audio files...")
    base_path = "audio_sources"

    audio_path = os.path.join(os.path.dirname(__file__), base_path, "speaker_1.wav")
    if not os.path.exists(audio_path):
        print(f"Warning: Example audio file not found at {audio_path}")
        return
    renderer.set_source_audio(src1, audio_path, gain=1.0)
    audio_path_2 = os.path.join(os.path.dirname(__file__), base_path, "speaker_2.wav")
    if not os.path.exists(audio_path_2):
        print(f"Warning: Example audio file not found at {audio_path_2}")
        return
    renderer.set_source_audio(src2, audio_path_2, gain=1.0)

    audio_path_bg = os.path.join(os.path.dirname(__file__), base_path, "foreground.wav")
    if not os.path.exists(audio_path_bg):
        print(f"Warning: Example audio file not found at {audio_path_bg}")
    else:
        renderer.set_source_audio(src_bg, audio_path_bg, gain=0.1)

    # 8. Render
    print("Starting Radiosity Rendering pipeline (ISM Order 2 + Radiosity)...")
    with PerformanceMonitor() as monitor:
        outputs, rirs = renderer.render(
            ism_order=2,
            rir_duration=1.5
        )
    save_performance_metrics(monitor, output_dir, "radiosity")

    # 9. Save Result
    mixed_audio = outputs[mic.name]
    rir = rirs[mic.name]

    if mixed_audio is not None:
        run_metrics_and_save(
            mixed_audio, rir, mic.name, mic_type, DEFAULT_SAMPLING_RATE,
            output_dir, "radiosity", save_rir_flag=save_rir_flag,
            save_audio_flag=save_audio_flag, save_acoustics_flag=save_acoustics_flag,
            save_psychoacoustics_flag=save_psychoacoustics_flag
        )
    else:
        print("Error: No audio output generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a radiosity simulation.")
    parser.add_argument(
        '--mic', type=str, default='mono', choices=['mono', 'ambisonic'],
        help="Microphone type."
    )
    parser.add_argument(
        '--output_dir', type=str, default='outputs/radiosity',
        help="Output directory."
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
    parser.set_defaults(save_audio=True, save_acoustics=True, save_psychoacoustics=False, save_mesh=True)
    args = parser.parse_args()
    main(mic_type=args.mic, output_dir=args.output_dir, save_rir_flag=args.save_rir,
         save_audio_flag=args.save_audio, save_acoustics_flag=args.save_acoustics,
         save_psychoacoustics_flag=args.save_psychoacoustics, save_mesh_flag=args.save_mesh)
