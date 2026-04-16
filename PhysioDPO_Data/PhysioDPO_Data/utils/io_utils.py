import json
import os
import numpy as np
from typing import List
from core.data_models import ScoredSequence, PreferencePair

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_scored_sequences(sequences: List[ScoredSequence], filepath: str):
    """Saves scored sequences to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for seq in sequences:
            data = {
                "id": seq.sequence.id,
                "sequence": seq.sequence.sequence,
                "plddt_mean": seq.structure.mean_plddt,
                "energy": seq.energy_score,
                "stability_score": seq.stability_score,
                "is_stable": seq.is_stable,
                "metadata": seq.metadata
            }
            f.write(json.dumps(data, cls=NumpyEncoder) + "\n")

def save_preference_pairs(pairs: List[PreferencePair], filepath: str):
    """Saves preference pairs to a JSONL file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for pair in pairs:
            f.write(json.dumps(pair.to_dict(), cls=NumpyEncoder) + "\n")

def load_scored_sequences(filepath: str) -> List[dict]:
    """Loads scored sequences metadata (not full objects) for analysis."""
    data = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data
