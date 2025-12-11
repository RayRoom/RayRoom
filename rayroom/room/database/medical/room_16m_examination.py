from .base import MedicalRoomBase
from ....room import objects


class MedicalRoom16MExamination(MedicalRoomBase):
    def __init__(self, mic_type='mono'):
        super().__init__(area=16, mic_type=mic_type)

    def _add_receiver(self, room):
        length, width = self.dimensions
        mic_pos = [length / 2, width / 2, 1.7]
        if self.mic_type == 'ambisonic':
            mic = objects.AmbisonicReceiver("AmbiMic", mic_pos, radius=0.02)
        else:
            mic = objects.Receiver("MonoMic", mic_pos, radius=0.15)
        room.add_receiver(mic)
        return mic

    def _add_furniture(self, room):
        length, width = self.dimensions
        
        # Core examination furniture
        exam_table = objects.ExaminingTable("ExamTable", [length * 0.5, width * 0.3, 0], rotation_z=90)
        room.add_furniture(exam_table)
        
        stool = objects.MedicalStool("Stool", [length * 0.5 - 0.5, width * 0.6, 0])
        room.add_furniture(stool)
        
        # Medical equipment
        ecg = objects.ECG("ECG", [0.3, width * 0.2, 0])
        room.add_furniture(ecg)
        
        scale = objects.WeighingScale("Scale", [0.2, width - 0.5, 0])
        room.add_furniture(scale)

        # Sanitation and storage
        sink = objects.Sink("Sink", [length - 0.6, 0.1, 0], rotation_z=180)
        room.add_furniture(sink)
        
        bin = objects.SquareBin("Bin", [length - 0.8, width - 0.5, 0])
        room.add_furniture(bin)
        
        v_cabinets = objects.VerticalCabinets("VerticalCabinets", [length - 0.5, width * 0.5, 0])
        room.add_furniture(v_cabinets)
        
        # Room fixtures
        door = objects.Door("Door", [0.5, 0, 0], rotation_z=0)
        room.add_furniture(door)
        
        # Personnel
        # Patient laying on the table
        patient_pos = [exam_table.position[0], exam_table.position[1], 0.8]
        patient = objects.LayingPerson("Patient", patient_pos, rotation_z=90)
        room.add_furniture(patient)

        # Doctor near the stool
        doctor_pos = [stool.position[0] - 0.5, stool.position[1], 0]
        doctor = objects.Person("Doctor", doctor_pos, rotation_z=45)
        room.add_furniture(doctor)

    def _add_sources(self, room):
        length, width = self.dimensions
        height = 3.0
        src1_pos = [length * 0.8, width * 0.8, 1.2]
        src1 = objects.Source("MedicalDeviceBeep", src1_pos, power=0.1)
        src2_pos = [length * 0.2, width * 0.2, 1.6]
        src2 = objects.Source("HumanSpeech", src2_pos, power=0.8)
        src_bg = objects.Source("Background HVAC", [0.1, 0.1, height - 0.1], power=0.2)
        room.add_source(src1)
        room.add_source(src2)
        room.add_source(src_bg)
        return {"src1": src1, "src2": src2, "src_bg": src_bg}
