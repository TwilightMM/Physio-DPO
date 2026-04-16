from abc import ABC, abstractmethod
from typing import List, Tuple
from .data_models import ProteinSequence, StructurePrediction, ScoredSequence, PreferencePair

class BaseGenerator(ABC):
    """Abstract base class for sequence generation (e.g., using LM)."""
    
    @abstractmethod
    def generate(self, n_samples: int, **kwargs) -> List[ProteinSequence]:
        """Generate n_samples protein sequences."""
        pass

    def unload(self):
        """Optional method to unload resources."""
        pass

class BaseFolder(ABC):
    """Abstract base class for structure prediction (e.g., AlphaFold, ESMFold)."""
    
    @abstractmethod
    def fold(self, sequences: List[ProteinSequence], **kwargs) -> List[StructurePrediction]:
        """Predict structures for a batch of sequences."""
        pass

class BaseScorer(ABC):
    """Abstract base class for scoring structures (e.g., pLDDT + Rosetta Energy)."""
    
    @abstractmethod
    def score(self, sequences: List[ProteinSequence], structures: List[StructurePrediction], **kwargs) -> List[ScoredSequence]:
        """Compute stability scores and energy for folded structures."""
        pass

class BasePairingStrategy(ABC):
    """Abstract base class for constructing preference pairs."""
    
    @abstractmethod
    def pair(self, scored_sequences: List[ScoredSequence], **kwargs) -> List[PreferencePair]:
        """Construct preference pairs from a pool of scored sequences."""
        pass
