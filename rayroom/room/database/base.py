from abc import abstractmethod


class Entry:

    @abstractmethod
    def create_room(self) -> tuple:
        raise NotImplementedError("Subclasses must implement this method")
