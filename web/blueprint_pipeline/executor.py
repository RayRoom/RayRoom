#!/usr/bin/env python3
"""
Executor for RayRoom pipelines from node-based blueprints
"""

import os
import sys
import time
import json
import shutil
import traceback
import numpy as np
from pathlib import Path
from scipy.io import wavfile

from rayroom.core.constants import DEFAULT_SAMPLING_RATE
from rayroom import (
    HybridRenderer, RadiosityRenderer, SpectralRenderer, RaytracingRenderer
)
from rayroom.room.database import (
    DemoRoom, TestBenchRoom, MedicalRoom8M, MedicalRoom12M
)
from rayroom.analytics.performance import PerformanceMonitor
from rayroom.analytics.acoustics import (
    calculate_clarity, calculate_drr, calculate_edt, calculate_rt60,
    schroeder_integration, calculate_loudness, calculate_sharpness, calculate_roughness
)

sys.path.insert(0, str(Path(__file__).parent.parent))


class BlueprintExecutor:
    """Executes a blueprint pipeline"""

    def __init__(self):
        self.node_results = {}
        self.room = None
        self.sources = {}
        self.receiver = None
        self.renderer = None

        # Create temporary directory for audio files in blueprint_pipeline directory
        blueprint_pipeline_dir = Path(__file__).parent
        temp_base = blueprint_pipeline_dir / 'tmp'
        temp_base.mkdir(exist_ok=True)

        # Create a unique subdirectory for this execution
        timestamp = int(time.time() * 1000)  # milliseconds
        self.temp_dir = temp_base / f'rayroom_blueprint_{timestamp}'
        self.temp_dir.mkdir(exist_ok=True)
        print(f"Created temporary directory: {self.temp_dir}")

    def __del__(self):
        """Clean up temporary directory"""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                print(f"Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(f"Warning: Could not clean up temp directory: {e}")

    def _resolve_audio_path(self, filename):
        """Resolve audio file path by checking multiple locations"""
        if not filename:
            return None

        path = Path(filename)

        # 1. Check exact path (absolute or relative to CWD)
        if path.exists():
            return path

        # Define base directories to search
        executor_dir = Path(__file__).parent
        project_root = executor_dir.parent.parent

        search_paths = [
            executor_dir / filename,
            project_root / filename,
            project_root / 'examples' / 'engines' / 'audio_sources' / filename,
            project_root / 'examples' / 'audio_sources' / filename,
        ]

        for search_path in search_paths:
            if search_path.exists():
                return search_path

        return None

    def _copy_audio_to_temp(self, source_path):
        """Copy an audio file to the temporary directory and return the temp path"""
        if not source_path or not os.path.exists(source_path):
            return None
        # Get the filename
        filename = os.path.basename(source_path)
        temp_path = self.temp_dir / filename
        # Copy the file if it doesn't already exist in temp
        if not temp_path.exists():
            try:
                shutil.copy2(source_path, temp_path)
                print(f"Copied audio file to temp: {source_path} -> {temp_path}")
            except Exception as e:
                print(f"Error copying audio file: {e}")
                return None
        return str(temp_path)

    def execute(self, blueprint):
        """Execute the blueprint pipeline"""
        try:
            # Build execution graph
            execution_order = self._topological_sort(blueprint)
            print(f"Execution order: {execution_order}")
            # Execute nodes in order
            for node_id in execution_order:
                node_data = next(n for n in blueprint['nodes'] if n['id'] == node_id)
                print(f"Executing node {node_id}: {node_data['type']}")
                self._execute_node(node_data, blueprint)
            # Get audio file path if available (only if audio was actually saved)
            audio_file_path = self.node_results.get('audio_file_path')
            result = {'success': True, 'message': 'Pipeline executed successfully'}
            # Only include audio file path if it was actually created
            if audio_file_path:
                # Make path relative to web directory for serving
                web_dir = Path(__file__).parent.parent
                try:
                    audio_path = Path(audio_file_path)
                    # Check if path is absolute or relative
                    if audio_path.is_absolute():
                        # Try to make it relative to web directory
                        try:
                            relative_path = audio_path.relative_to(web_dir)
                            full_path = web_dir / relative_path
                        except ValueError:
                            # Can't make relative, use absolute
                            full_path = audio_path
                            relative_path = audio_path
                    else:
                        # Already relative, construct full path to check existence
                        full_path = web_dir / audio_path
                        relative_path = audio_path
                    # Check if file actually exists
                    if full_path.exists():
                        result['audioFile'] = str(relative_path)
                        print(f"Returning audio file path: {result['audioFile']} (full path: {full_path})")
                    else:
                        print(f"Warning: Audio file path provided but file doesn't exist: {full_path}")
                except Exception as e:
                    print(f"Warning: Could not process audio file path: {e}")
                    traceback.print_exc()
            return result
        except Exception as e:
            traceback.print_exc()
            return {'success': False, 'error': str(e)}

    def _topological_sort(self, blueprint):
        """Sort nodes by dependencies"""
        nodes = {n['id']: n for n in blueprint['nodes']}
        connections = blueprint.get('connections', [])
        # Build dependency graph
        dependencies = {node_id: [] for node_id in nodes.keys()}
        for conn in connections:
            to_node = conn['to']['nodeId']
            from_node = conn['from']['nodeId']
            if to_node not in dependencies:
                dependencies[to_node] = []
            dependencies[to_node].append(from_node)
        # Topological sort
        result = []
        visited = set()
        temp_visited = set()

        def visit(node_id):
            if node_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving node {node_id}")
            if node_id in visited:
                return
            temp_visited.add(node_id)
            for dep in dependencies.get(node_id, []):
                visit(dep)
            temp_visited.remove(node_id)
            visited.add(node_id)
            result.append(node_id)
        for node_id in nodes.keys():
            if node_id not in visited:
                visit(node_id)
        return result

    def _execute_node(self, node_data, blueprint):
        """Execute a single node"""
        node_type = node_data['type']
        node_id = node_data['id']
        params = node_data.get('params', {})
        if node_type == 'room':
            self._execute_room_node(params)
        elif node_type == 'renderer':
            self._execute_renderer_node(params, blueprint)
        elif node_type == 'audio':
            self._execute_audio_node(node_id, params)
        elif node_type == 'metrics':
            self._execute_metrics_node(node_id, params, blueprint)
        elif node_type == 'output':
            self._execute_output_node(params, blueprint)

    def _execute_room_node(self, params):
        """Create room from database"""
        room_type = params.get('roomType', 'DemoRoom')
        mic_type = params.get('micType', 'mono')
        room_classes = {
            'DemoRoom': DemoRoom,
            'TestBenchRoom': TestBenchRoom,
            'MedicalRoom8M': MedicalRoom8M,
            'MedicalRoom12M': MedicalRoom12M,
        }
        if room_type not in room_classes:
            raise ValueError(f"Unknown room type: {room_type}")
        room_class = room_classes[room_type]
        self.room, self.sources, self.receiver = room_class(mic_type=mic_type).create_room()
        self.node_results['room'] = self.room

    def _execute_source_node(self, node_id, params, blueprint):
        """Add or configure source"""
        if not self.room:
            raise ValueError("Room must be created before sources")
        name = params.get('name', f'src_{node_id}')
        position_str = params.get('position', '2,1,1.5')
        position = [float(x.strip()) for x in position_str.split(',')]
        from rayroom.room.objects import Source
        source = Source(name, position, radius=0.1)
        self.room.add_source(source)
        self.sources[name] = source
        self.node_results[f'source_{node_id}'] = source

    def _execute_receiver_node(self, params, blueprint):
        """Add or configure receiver"""
        if not self.room:
            raise ValueError("Room must be created before receiver")
        name = params.get('name', 'mic')
        position_str = params.get('position', '1,0.5,1.5')
        position = [float(x.strip()) for x in position_str.split(',')]
        mic_type = params.get('micType', 'mono')
        if mic_type == 'ambisonic':
            from rayroom.room.objects import AmbisonicReceiver
            receiver = AmbisonicReceiver(name, position, radius=0.02)
        else:
            from rayroom.room.objects import Receiver
            receiver = Receiver(name, position, radius=0.15)
        self.room.add_receiver(receiver)
        self.receiver = receiver
        self.node_results['receiver'] = receiver

    def _execute_renderer_node(self, params, blueprint):
        """Create and configure renderer"""
        if not self.room:
            raise ValueError("Room must be created before renderer")
        renderer_type = params.get('rendererType', 'HybridRenderer')
        fs = int(params.get('fs', DEFAULT_SAMPLING_RATE))
        temperature = float(params.get('temperature', 20.0))
        humidity = float(params.get('humidity', 50.0))
        renderer_classes = {
            'HybridRenderer': HybridRenderer,
            'RadiosityRenderer': RadiosityRenderer,
            'SpectralRenderer': SpectralRenderer,
            'RaytracingRenderer': RaytracingRenderer,
        }
        if renderer_type not in renderer_classes:
            raise ValueError(f"Unknown renderer type: {renderer_type}")
        renderer_class = renderer_classes[renderer_type]
        if renderer_type == 'RadiosityRenderer':
            self.renderer = renderer_class(self.room, fs=fs)
        elif renderer_type in ['HybridRenderer', 'SpectralRenderer']:
            self.renderer = renderer_class(self.room, fs=fs, temperature=temperature, humidity=humidity)
        else:
            self.renderer = renderer_class(self.room, fs=fs)
        # Assign audio files to sources based on connections to renderer
        # Find all audio nodes connected to the renderer
        renderer_node_id = next(n['id'] for n in blueprint['nodes'] if n['type'] == 'renderer')
        print(f"Available sources: {list(self.sources.keys())}")
        print(f"Receiver: {self.receiver.name if self.receiver else 'None'}")
        # Map audio nodes to sources by sourceName parameter
        audio_assigned = False
        for conn in blueprint.get('connections', []):
            from_node = next((n for n in blueprint['nodes'] if n['id'] == conn['from']['nodeId']), None)
            to_node = next((n for n in blueprint['nodes'] if n['id'] == conn['to']['nodeId']), None)
            # Audio nodes connected to renderer
            if (from_node and from_node['type'] == 'audio' and
                    to_node and to_node['id'] == renderer_node_id):
                source_name = from_node['params'].get('sourceName', '')
                audio_file = from_node['params'].get('file', '')
                gain = float(from_node['params'].get('gain', 1.0))
                print(f"Processing audio node: file={audio_file}, sourceName={source_name}, gain={gain}")
                if audio_file and source_name:
                    if source_name in self.sources:
                        print(f"Checking audio file: {audio_file}")
                        # Resolve audio file path
                        resolved_path = self._resolve_audio_path(audio_file)

                        if resolved_path:
                            # Copy to temp
                            temp_audio_path = self._copy_audio_to_temp(str(resolved_path))
                            if temp_audio_path:
                                print(f"Assigning audio file {temp_audio_path} to source {source_name}")
                                self.renderer.set_source_audio(
                                    self.sources[source_name],
                                    temp_audio_path,
                                    gain=gain
                                )
                                audio_assigned = True
                            else:
                                raise ValueError(f"Could not copy audio file to temp directory: {resolved_path}")
                        else:
                            # Error if not found - this will be caught and shown in UI
                            raise FileNotFoundError(
                                f"Audio file not found: {audio_file} (searched in common locations)"
                            )
                    else:
                        print(
                            f"Warning: Source '{source_name}' not found in room. "
                            f"Available sources: {list(self.sources.keys())}"
                        )
        if not audio_assigned:
            print("Warning: No audio files were assigned to sources. Renderer will proceed without audio.")
        # Render
        ism_order = int(params.get('ismOrder', 2))
        n_rays = int(params.get('nRays', 20000))
        max_hops = int(params.get('maxHops', 50))
        rir_duration = float(params.get('rirDuration', 1.5))
        try:
            with PerformanceMonitor() as monitor:
                if renderer_type == 'RadiosityRenderer':
                    outputs, rirs = self.renderer.render(
                        ism_order=ism_order,
                        rir_duration=rir_duration
                    )
                elif renderer_type in ['HybridRenderer', 'SpectralRenderer']:
                    outputs, _, rirs = self.renderer.render(
                        n_rays=n_rays,
                        max_hops=max_hops,
                        rir_duration=rir_duration,
                        ism_order=ism_order
                    )
                else:
                    outputs, rirs = self.renderer.render(
                        n_rays=n_rays,
                        max_hops=max_hops,
                        rir_duration=rir_duration
                    )
            # Check if outputs/rirs are empty or contain only None values
            # Renderers return {rx.name: None} when no audio is assigned, not empty dict
            has_valid_outputs = outputs and any(v is not None for v in outputs.values())
            if not rirs:
                print("Warning: Renderer produced no RIRs")
            # If both are empty/invalid, that's an error
            if not has_valid_outputs and not rirs:
                raise ValueError("Renderer produced empty outputs and RIRs - rendering failed completely")
            # If outputs are invalid but we have RIRs, that's okay (no audio assigned)
            if not has_valid_outputs:
                print("Note: No audio outputs (likely no audio files assigned), but RIRs are available")
                outputs = {}  # Create empty dict to avoid errors downstream
            self.node_results['renderer_output'] = outputs
            self.node_results['renderer_rir'] = rirs
            self.node_results['performance'] = {
                'runtime_s': monitor.runtime_s,
                'peak_memory_mb': monitor.peak_memory_mb
            }
        except Exception as e:
            print(f"Error during rendering: {e}")
            traceback.print_exc()
            raise

    def _execute_audio_node(self, node_id, params):
        """Handle audio input node"""
        # In a real implementation, files would be uploaded and stored
        # For now, just store the file path
        audio_file = params.get('file', '')
        self.node_results[f'audio_{node_id}'] = {'file': audio_file}

    def _execute_metrics_node(self, node_id, params, blueprint):
        """Compute metrics"""
        outputs = self.node_results.get('renderer_output', {})
        rirs = self.node_results.get('renderer_rir', {})
        if not outputs or not rirs:
            raise ValueError(
                f"Renderer must be executed before metrics. "
                f"Outputs: {bool(outputs)}, RIRs: {bool(rirs)}"
            )
        # Try to find the receiver name - check common names
        receiver_name = None
        if self.receiver:
            receiver_name = self.receiver.name
        else:
            # Try common receiver names
            for name in ['MonoMic', 'AmbiMic', 'mic', 'receiver']:
                if name in outputs:
                    receiver_name = name
                    break
        if not receiver_name:
            # Use first available receiver
            receiver_name = list(outputs.keys())[0] if outputs else None
        if not receiver_name:
            raise ValueError(f"No receiver found. Available outputs: {list(outputs.keys())}")
        mixed_audio = outputs.get(receiver_name)
        rir = rirs.get(receiver_name)
        if mixed_audio is None or rir is None:
            raise ValueError(
                f"No audio or RIR available for receiver '{receiver_name}'. "
                f"Available outputs: {list(outputs.keys())}, "
                f"RIRs: {list(rirs.keys())}"
            )
        metrics = {}
        if params.get('acoustic', True):
            # Compute acoustic metrics
            if rir.ndim > 1:
                rir_mono = rir[:, 0]
            else:
                rir_mono = rir
            sch_db = schroeder_integration(rir_mono)
            metrics['edt'] = calculate_edt(sch_db, DEFAULT_SAMPLING_RATE)
            metrics['rt60'] = calculate_rt60(sch_db, DEFAULT_SAMPLING_RATE)
            metrics['c50'] = calculate_clarity(rir_mono, DEFAULT_SAMPLING_RATE, 50)
            metrics['c80'] = calculate_clarity(rir_mono, DEFAULT_SAMPLING_RATE, 80)
            metrics['drr'] = calculate_drr(rir_mono, DEFAULT_SAMPLING_RATE)
        if params.get('psychoacoustic', False):
            # Compute psychoacoustic metrics
            audio_for_metrics = mixed_audio[:, 0] if mixed_audio.ndim > 1 else mixed_audio
            metrics['loudness'], _ = calculate_loudness(audio_for_metrics, DEFAULT_SAMPLING_RATE)
            metrics['sharpness'] = calculate_sharpness(audio_for_metrics, DEFAULT_SAMPLING_RATE)
            roughness_array, _ = calculate_roughness(audio_for_metrics, DEFAULT_SAMPLING_RATE)
            metrics['roughness'] = (
                np.mean(roughness_array)
                if isinstance(roughness_array, np.ndarray)
                else roughness_array
            )
        if params.get('performance', False):
            metrics['performance'] = self.node_results.get('performance', {})
        self.node_results[f'metrics_{node_id}'] = metrics

    def _execute_output_node(self, params, blueprint):
        """Save outputs"""
        output_dir = params.get('outputDir', 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        outputs = self.node_results.get('renderer_output', {})
        rirs = self.node_results.get('renderer_rir', {})
        receiver_name = self.receiver.name if self.receiver else 'mic'
        mixed_audio = outputs.get(receiver_name)
        rir = rirs.get(receiver_name)
        # Check if we actually have valid audio (not None and not empty)
        has_valid_audio = (mixed_audio is not None and
                           isinstance(mixed_audio, np.ndarray) and
                           mixed_audio.size > 0)
        if not has_valid_audio:
            print(f"Note: No valid audio to save for receiver '{receiver_name}'. "
                  f"Audio outputs: {outputs}, Mixed audio type: {type(mixed_audio)}")
        mic_type = 'mono' if self.receiver and not hasattr(self.receiver, 'order') else 'ambisonic'
        audio_file_path = None
        if params.get('saveAudio', True) and has_valid_audio:
            output_filename = f"output_{mic_type}.wav"
            # Make output_dir relative to web directory if not absolute
            if not os.path.isabs(output_dir):
                web_dir = Path(__file__).parent.parent
                output_path = web_dir / output_dir / output_filename
            else:
                output_path = Path(output_dir) / output_filename
            # Create directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            mixed_audio = mixed_audio / np.max(np.abs(mixed_audio))
            if mic_type == 'ambisonic':
                wavfile.write(str(output_path), DEFAULT_SAMPLING_RATE, mixed_audio.astype(np.float32))
            else:
                wavfile.write(str(output_path), DEFAULT_SAMPLING_RATE, (mixed_audio * 32767).astype(np.int16))
            # Store the audio file path relative to web directory for the audio player
            web_dir = Path(__file__).parent.parent
            try:
                audio_file_path = str(output_path.relative_to(web_dir))
            except ValueError:
                # If path is not relative to web_dir, use absolute path
                audio_file_path = str(output_path)
            self.node_results['audio_file_path'] = audio_file_path
            print(f"Saved audio file to: {output_path} (relative path: {audio_file_path})")
        if params.get('saveRIR', False) and rir is not None:
            rir_filename = f"rir_{mic_type}.wav"
            rir_path = os.path.join(output_dir, rir_filename)
            rir = rir / np.max(np.abs(rir))
            if mic_type == 'ambisonic':
                wavfile.write(rir_path, DEFAULT_SAMPLING_RATE, rir.astype(np.float32))
            else:
                wavfile.write(rir_path, DEFAULT_SAMPLING_RATE, (rir * 32767).astype(np.int16))
        if params.get('saveMetrics', True):
            # Find metrics
            metrics = None
            for key, value in self.node_results.items():
                if key.startswith('metrics_'):
                    metrics = value
                    break
            if metrics:
                metrics_path = os.path.join(output_dir, 'metrics.json')
                with open(metrics_path, 'w') as f:
                    json.dump(metrics, f, indent=4)
        if params.get('saveMesh', False) and self.room:
            mesh_path = os.path.join(output_dir, 'room_mesh.obj')
            self.room.save_mesh(mesh_path)
