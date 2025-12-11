from .base import MedicalRoomBase
from ....room import objects


class MedicalRoom16MConsulting(MedicalRoomBase):
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
        
        # Consulting area
        desk = objects.Desk("Desk", [length * 0.3, width * 0.8, 0], rotation_z=180)
        room.add_furniture(desk)
        
        doctor_chair = objects.Chair("DoctorChair", [length * 0.3, width * 0.8 - 0.6, 0], rotation_z=0)
        room.add_furniture(doctor_chair)

        patient_chair = objects.Chair("PatientChair", [length * 0.3, width * 0.8 + 0.6, 0], rotation_z=180)
        room.add_furniture(patient_chair)

        computer = objects.iMac("Computer", [desk.position[0] - 0.2, desk.position[1], desk.height], rotation_z=180)
        room.add_furniture(computer)

        # Sanitation and storage
        sink = objects.Sink("Sink", [length - 0.6, 0.1, 0], rotation_z=180)
        room.add_furniture(sink)
        
        bin = objects.SquareBin("Bin", [length - 0.8, width - 0.5, 0])
        room.add_furniture(bin)

        h_cabinets = objects.HorizontalCabinets("WallCabinets", [length * 0.5, 0.1, 1.5], rotation_z=90)
        room.add_furniture(h_cabinets)
        
        # Room fixtures
        door = objects.Door("Door", [0.5, 0, 0], rotation_z=0)
        room.add_furniture(door)
        
        # Personnel
        # Doctor sitting at the desk
        doctor_pos = [doctor_chair.position[0], doctor_chair.position[1], 0.45]
        doctor = objects.SittingPerson("Doctor", doctor_pos, rotation_z=0)
        room.add_furniture(doctor)
        
        # Patient sitting in the chair
        patient_pos = [patient_chair.position[0], patient_chair.position[1], 0.45]
        patient = objects.SittingPerson("Patient", patient_pos, rotation_z=180)
        room.add_furniture(patient)

    def _add_sources(self, room):
        length, width = self.dimensions
        height = 3.0
        # Source from the doctor
        src1_pos = [length * 0.3, width * 0.8 - 0.4, 1.2]
        src1 = objects.Source("DoctorSpeech", src1_pos, power=0.8)
        
        # Source from the patient
        src2_pos = [length * 0.3, width * 0.8 + 0.4, 1.2]
        src2 = objects.Source("PatientSpeech", src2_pos, power=0.8)
        
        src_bg = objects.Source("Background HVAC", [0.1, 0.1, height - 0.1], power=0.2)
        room.add_source(src1)
        room.add_source(src2)
        room.add_source(src_bg)
        return {"src1": src1, "src2": src2, "src_bg": src_bg}
