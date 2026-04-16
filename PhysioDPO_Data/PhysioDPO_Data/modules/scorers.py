from typing import List, Tuple
from core.interfaces import BaseScorer
from core.data_models import ProteinSequence, StructurePrediction, ScoredSequence
import numpy as np
class PhysioScorer(BaseScorer):
    """
    Scores sequences based on pLDDT.
    In a full production environment, this would also call PyRosetta for Energy (REU).
    Here we focus on pLDDT as the primary stability metric.
    """
    
    def __init__(self, 
                 plddt_threshold: float = 70.0, 
                 plddt_low_threshold: float = 50.0):
        self.plddt_threshold = plddt_threshold
        self.plddt_low_threshold = plddt_low_threshold

    def score(self, sequences: List[ProteinSequence], structures: List[StructurePrediction], **kwargs) -> List[ScoredSequence]:
        scored_sequences = []
        
        # Map structures by ID
        struct_map = {s.sequence_id: s for s in structures}
        
        for seq in sequences:
            struct = struct_map.get(seq.id)
            if not struct:
                continue
            
            # Real Scoring Logic:
            # 1. Stability based on pLDDT (Mean pLDDT)
            # 2. Energy: In this environment, we default to 0.0 as we don't have Rosetta.
            
            energy = 0.0 
            
            is_stable = False
            
            if struct.mean_plddt >= self.plddt_threshold:
                is_stable = True
            elif struct.mean_plddt < self.plddt_low_threshold:
                is_stable = False
            
            stability_score = struct.mean_plddt

            scored_sequences.append(ScoredSequence(
                sequence=seq,
                structure=struct,
                energy_score=energy,
                stability_score=stability_score,
                is_stable=is_stable,
                metadata={
                    "plddt_threshold": self.plddt_threshold,
                    "scorer": "plddt_only"
                }
            ))
            
        return scored_sequences
