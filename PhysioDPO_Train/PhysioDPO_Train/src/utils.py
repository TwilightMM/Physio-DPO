import torch
from dataclasses import dataclass
from typing import List, Dict, Any
from transformers import DataCollatorWithPadding

@dataclass
class PhysioDataCollator:
    """
    Custom Collator to ensure energy_gap is properly stacked into Tensor
    while preserving correct dtypes for other fields (especially input_ids as Long)
    """
    base_collator: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        # 1. Extract energy_gap (keep it separate to avoid dtype contamination)
        energy_gaps = None
        if features and "energy_gap" in features[0]:
            energy_gaps = [f.pop("energy_gap", 0.0) for f in features]
            energy_gaps = torch.tensor(energy_gaps, dtype=torch.float32)

        # 2. Call standard DPO collator to process input_ids etc.
        batch = self.base_collator(features)

        # 3. Ensure input_ids are Long type (critical for embedding layer)
        for key in batch:
            if "input_ids" in key or "labels" in key or "attention_mask" in key:
                if isinstance(batch[key], torch.Tensor):
                    if "attention_mask" in key:
                        batch[key] = batch[key].long()
                    elif batch[key].dtype == torch.float32 or batch[key].dtype == torch.float16:
                        # Force conversion to Long for input_ids and labels
                        batch[key] = batch[key].long()

        # 4. Put energy_gap back
        if energy_gaps is not None:
            batch["energy_gap"] = energy_gaps
        
        return batch

def find_all_linear_names(model):
    """
    Automatically find all linear layers in ProGen2 for LoRA
    """
    import bitsandbytes as bnb
    cls = bnb.nn.Linear4bit 
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls) or isinstance(module, torch.nn.Linear):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # Usually skip head
        lora_module_names.remove('lm_head')
    return list(lora_module_names)