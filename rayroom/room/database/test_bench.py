from rayroom.room.database.base import Entry
from rayroom.room import objects, Room, get_material
from rayroom.core.constants import DEFAULT_SAMPLING_RATE


class TestBenchRoom(Entry):

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
        Creates a huge room with all available objects.

        Returns
        -------
        tuple
            A tuple containing:
            - room (Room): The configured room object.
            - sources (dict): A dictionary of sources.
            - mic (Receiver or AmbisonicReceiver): The receiver object.
        """
        print("Creating huge room with all objects (25m x 20m x 5m)...")
        room = Room.create_shoebox([25, 20, 5], materials={
            "floor": get_material("concrete"),
            "ceiling": get_material("plaster"),
            "walls": get_material("brick")
        }, fs=DEFAULT_SAMPLING_RATE)

        # Add Receiver
        mic_pos = [12.5, 10, 1.7]
        if self.mic_type == 'ambisonic':
            mic = objects.AmbisonicReceiver("AmbiMic", mic_pos)
        else:
            mic = objects.Receiver("MonoMic", mic_pos)
        room.add_receiver(mic)

        # === Add All Furniture and Objects ===
        all_objects = {
            # Living Room Zone (front-left corner)
            "ThreeSeatCouch": objects.ThreeSeatCouch("Couch1", [1, 4, 0], rotation_z=-90),
            "TwoSeatCouch": objects.TwoSeatCouch("Couch2", [4.25, 4, 0], rotation_z=90),
            "OneSeatCouch": objects.OneSeatCouch("Armchair", [2.5, 5.6, 0], rotation_z=180),
            "CoffeeTable": objects.CoffeeTable("CoffeeTbl", [2.7, 4, 0]),
            "SquareCarpet": objects.SquareCarpet("Carpet1", [2.6, 4, 0.01]),
            "DiningTable": objects.DiningTable("DiningTbl", [8, 6, 0]),
            "Chair": objects.Chair("Chair1", [7.5, 5, 0]),
            "Desk": objects.Desk("Desk1", [2, 10, 0], rotation_z=15),
            "Subwoofer": objects.Subwoofer("Sub1", [1, 1, 0], rotation_z=90),
            "FloorstandingSpeaker": objects.FloorstandingSpeaker("SpeakerL", [1, 6, 0], rotation_z=90),
            "TV": objects.TV("TV1", [4, 1, 0.8], rotation_z=0),
            "FloatingTVShelf": objects.FloatingTVShelf("TVShelf", [4, 1, 0.5]),

            # Wall Objects (placed on various walls)
            "Window": objects.Window("Window1", [0, 5, 2.5], rotation_z=90),
            "DoubleRectangleWindow": objects.DoubleRectangleWindow("Window2", [10, 0, 2.5]),
            "SquareWindow": objects.SquareWindow("Window3", [25, 8, 2], rotation_z=-90),
            "Painting": objects.Painting("Art1", [15, 0, 3]),
            "WallShelf": objects.WallShelf("Shelf1", [2, 20, 3], rotation_z=180),
            "KitchenCabinet": objects.KitchenCabinet("K-Cabinet", [20, 0, 2.5]),
            "Clock": objects.Clock("Clock1", [12.5, 0, 4]),
            "Door": objects.Door("Door1", [15, 0, 0]),
            "ACWallUnit": objects.ACWallUnit("AC1", [25, 10, 4], rotation_z=-90),

            # Electronics Zone (on/near desk)
            "Smartphone": objects.Smartphone("Phone1", [2.1, 12.1, 0.75]),
            "Tablet": objects.Tablet("Tablet1", [2.4, 12.2, 0.75]),
            "EchoDot5": objects.EchoDot5("Echo5", [1, 12, 0.75]),
            "EchoDot2": objects.EchoDot2("Echo2", [1, 12.3, 0.75]),
            "EchoShow5": objects.EchoShow5("EchoShow", [1.5, 12.5, 0.75]),
            "GoogleNestMini": objects.GoogleNestMini("NestMini", [1, 11.7, 0.75]),
            "AmazonEcho2": objects.AmazonEcho2("EchoGen2", [1, 11.4, 0.75]),
            "CRTMonitor": objects.CRTMonitor("CRT", [2.2, 9.5, 0.75], rotation_z=90),
            "LCDMonitor": objects.LCDMonitor("LCD", [1.8, 7.5, 0.75], rotation_z=90),
            "iMac": objects.iMac("iMac1", [2, 12.8, 0.75], rotation_z=90),
            "Laptop": objects.Laptop("Laptop1", [12, 10, 0], rotation_z=90),
            "Printer": objects.Printer("Printer1", [8, 10, 0], rotation_z=90),
            "StackOfPaper": objects.StackOfPaper("Paper1", [1, 13.5, 0]),

            # Hospital Zone (back-right corner)
            "HospitalBed": objects.HospitalBed("HospitalBed", [22, 17, 0]),
            "ExaminingTable": objects.ExaminingTable("ExamTbl", [18, 17, 0]),
            "DentalChair": objects.DentalChair("DentistChair", [15, 17, 0], rotation_z=15),
            "MedicalStool": objects.MedicalStool("Stool1", [16, 16, 0], rotation_z=15),
            "HorizontalCabinets": objects.HorizontalCabinets("HCabinet", [24, 15, 1]),
            "VerticalCabinets": objects.VerticalCabinets("VCabinet", [24, 14, 0]),
            "Sink": objects.Sink("Sink1", [24, 12, 0]),
            "Wheelchair": objects.Wheelchair("WC1", [20, 15, 0], rotation_z=90),
            "Walker": objects.Walker("Walker1", [21, 15, 0]),
            "Defibrillator": objects.Defibrillator("Defib1", [18, 15, 0]),
            "WeighingScale": objects.WeighingScale("Scale1", [17, 15, 0]),
            "MRIScanner": objects.MRIScanner("MRI", [18, 10, 0]),
            "Ventilator": objects.Ventilator("Vent1", [20, 18, 0]),
            "UltrasoundMachine": objects.UltrasoundMachine("Ultra1", [17, 18, 0]),
            "ECG": objects.ECG("ECG1", [16, 18, 0]),
            "OperatingRoomLight": objects.OperatingRoomLight("OR-Light", [19, 17, 5]),

            # Miscellaneous
            "Person": objects.Person("Bob", [10, 10, 0]),
            "RoundBin": objects.RoundBin("Bin1", [1, 14, 0]),
            "SquareBin": objects.SquareBin("Bin2", [14, 1, 0]),
            "CeilingFan": objects.CeilingFan("Fan1", [12.5, 10, 5], rotation_z=15),
            "TallFanOnFoot": objects.TallFanOnFoot("TallFan", [1, 18, 0], rotation_z=90),
            "SmallFanOnFoot": objects.SmallFanOnFoot("SmallFan", [14, 18, 0], rotation_z=90),
            "TissueBox": objects.TissueBox("Tissues", [2.1, 12, 0.75+0.3]),
        }

        for name, obj in all_objects.items():
            room.add_furniture(obj)

        # Define Sources
        src1 = objects.Source("Source1", [5, 5, 2])
        src2 = objects.Source("Source2", [20, 15, 1.8])
        src_bg = objects.Source("Background Noise", [1, 1, 4.5], power=0.5)
        room.add_source(src1)
        room.add_source(src2)
        room.add_source(src_bg)

        sources = {"src1": src1, "src2": src2, "src_bg": src_bg}

        return room, sources, mic
