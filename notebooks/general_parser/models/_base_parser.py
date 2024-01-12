from abc import ABC, abstractmethod

class AbstractParser(ABC):
    @abstractmethod
    def parse_amr(self,text):
        pass
        
    def generate_text(self, amr):
        pass