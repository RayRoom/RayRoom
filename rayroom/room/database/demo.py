from rayroom.room.database.base import Entry
from rayroom.room import objects, Room, get_material
from rayroom.core.constants import DEFAULT_SAMPLING_RATE


class DemoRoom(Entry):

    def __init__(self, mic_type='mono'):
        """
        Parameters
        ----------
        mic_type : str, optional
            The type of microphone to use, by default 'mono'.
            Can be 'mono' or 'ambisonic'.
        """
        self.mic_type = mic_type

    def create_room(self) -> tuple:
        """
        Creates a standard demo room with furniture, sources, and a receiver.

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
        mic_pos = [1.2, 0.25, 1.7]
        if self.mic_type == 'ambisonic':
            print("Using Ambisonic Receiver.")
            mic = objects.AmbisonicReceiver("AmbiMic", mic_pos, radius=0.02)
        else:
            print("Using Mono Receiver.")
            mic = objects.Receiver("MonoMic", mic_pos, radius=0.15)
        room.add_receiver(mic)

        # 4. Add Furniture
        person_1 = objects.Person(
            "Person 1",
            [0.5, 1.5, 0],
            rotation_z=90,
            height=1.7,
            width=0.5,
            depth=0.3,
            material_name="human"
        )
        room.add_furniture(person_1)

        person_2 = objects.Person(
            "Person 2",
            [3.5, 1.5, 0],
            rotation_z=-90,
            height=1.7,
            width=0.5,
            depth=0.3,
            material_name="human"
        )
        room.add_furniture(person_2)

        chair = objects.Chair("Chair", [0.5, 0.65, 0], rotation_z=-45, material_name="wood")
        room.add_furniture(chair)

        couch = objects.ThreeSeatCouch("Couch", [2.9, 0.6, 0], rotation_z=5, material_name="fabric")
        room.add_furniture(couch)

        coffee_table = objects.CoffeeTable("CoffeeTable", [2.5, 1.5, 0], rotation_z=5, material_name="wood")
        room.add_furniture(coffee_table)

        door = objects.Door("Door 1", [0, 1.5, 0], rotation_z=90, material_name="wood")
        room.add_furniture(door)
        window_2 = objects.DoubleRectangleWindow("Double window 2", [3.0, 0, 1.5], material_name="glass")
        room.add_furniture(window_2)

        wall_shelf = objects.WallShelf("Wall shelf", [1.0, 0.15, 1.5], material_name="wood")
        room.add_furniture(wall_shelf)

        amazon_echo = objects.AmazonEcho2("Amazon Echo 2", [1.2, 0.15, 1.53], material_name="fabric", resolution=10)
        room.add_furniture(amazon_echo)

        # 5. Define Sources
        src1 = objects.Source("Speaker 1", [0.7, 1.5, 1.5], power=1.0, orientation=[1, 0, 0], directivity="cardioid")
        src2 = objects.Source("Speaker 2", [3.3, 1.5, 1.5], power=1.0, orientation=[-1, 0, 0], directivity="cardioid")
        src_bg = objects.Source("Background Noise", [0.1, 0.1, 2.4], power=0.5)

        room.add_source(src1)
        room.add_source(src2)
        room.add_source(src_bg)

        sources = {
            "src1": src1,
            "src2": src2,
            "src_bg": src_bg
        }

        return room, sources, mic
