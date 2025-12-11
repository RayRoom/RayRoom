from .base import MedicalRoomBase
from ....room import objects


class MedicalRoom6M(MedicalRoomBase):
    def __init__(self, mic_type='mono'):
        super().__init__(area=6, mic_type=mic_type)

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
        exam_table = objects.ExaminingTable("ExamTable", [length / 2, width * 0.3, 0], rotation_z=90)
        room.add_furniture(exam_table)
        stool = objects.MedicalStool("Stool", [length / 2 - 0.5, width * 0.6, 0])
        room.add_furniture(stool)
        sink = objects.Sink("Sink", [length - 0.6, 0.1, 0], rotation_z=180)
        room.add_furniture(sink)
        bin = objects.SquareBin("Bin", [length - 0.8, width - 0.5, 0])
        room.add_furniture(bin)
        door = objects.Door("Door", [0.5, 0, 0], rotation_z=0)
        room.add_furniture(door)

        # Add a person behind src1
        src1_pos = [length * 0.8, width * 0.8, 1.2]
        person1_pos = [src1_pos[0] + 0.5, src1_pos[1], 0]
        person1 = objects.Person("Person1", person1_pos, rotation_z=180)
        room.add_furniture(person1)

        # Add a person behind src2
        src2_pos = [length * 0.2, width * 0.2, 1.6]
        person_pos = [src2_pos[0] - 0.5, src2_pos[1], 0]
        person = objects.Person("Person", person_pos, rotation_z=0)
        room.add_furniture(person)

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
