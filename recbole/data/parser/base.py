import os
import abc

class BaseParser(abc.ABC):
    """
    Base class for all dataset parsers.
    Parsers are strictly responsible for converting raw datasets into the unified internal format (e.g., .inter files).
    They are decoupled from downstream tasks (e.g., sequential, CTR).
    """
    def __init__(self, raw_data_path, output_path):
        self.raw_data_path = raw_data_path
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)

    @abc.abstractmethod
    def parse(self):
        """
        Parse the raw data and save it into the unified format under self.output_path.
        Must be implemented by dataset-specific subclasses.
        """
        pass
