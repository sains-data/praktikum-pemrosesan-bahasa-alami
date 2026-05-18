from abc import ABC, abstractmethod


class ChatEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Muat bobot ke memori."""

    @abstractmethod
    def reply(self, message: str, **kwargs) -> str:
        """Hasilkan jawaban untuk satu pesan pengguna."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass
