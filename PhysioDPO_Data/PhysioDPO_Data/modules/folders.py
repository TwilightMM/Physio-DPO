import torch
import numpy as np
import os
from typing import List, Optional
from transformers import AutoTokenizer, EsmForProteinFolding
from core.interfaces import BaseFolder
from core.data_models import ProteinSequence, StructurePrediction


os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

class ESMFoldFolder(BaseFolder):
    def __init__(self, model_name: str = "facebook/esmfold_v1", device: Optional[str] = None, chunk_size: int = 32):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        
        print(f"Loading ESMFold model: {model_name} on {self.device}...")
        print("Downloading ESMFold files (this may take several minutes for first download)...")
        
        cache_dir = os.path.join(os.getcwd(), 'physio_dpo', 'models')
        os.makedirs(cache_dir, exist_ok=True)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                local_files_only=False
            )
            print("Tokenizer loaded successfully!")
        except Exception as e:
            print(f"Failed to load from mirror, trying original HuggingFace site...")
            original_endpoint = os.environ.get('HF_ENDPOINT')
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    local_files_only=False
                )
                print("Tokenizer loaded from original site!")
            except Exception as e2:
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                raise Exception(f"Failed to load tokenizer: {e2}")
        
        print("Using float32 for ESMFold to ensure numerical stability.")
        dtype = torch.float32
        
        print("Downloading model files (this may take a while, please wait)...")
        try:
            self.model = EsmForProteinFolding.from_pretrained(
                model_name, 
                low_cpu_mem_usage=True, 
                cache_dir=cache_dir,
                dtype=dtype,
                local_files_only=False
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model from mirror, trying original HuggingFace site...")
            original_endpoint = os.environ.get('HF_ENDPOINT')
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
            try:
                self.model = EsmForProteinFolding.from_pretrained(
                    model_name, 
                    low_cpu_mem_usage=True, 
                    cache_dir=cache_dir,
                    dtype=dtype,
                    local_files_only=False
                )
                print("Model loaded from original site!")
            except Exception as e2:
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                raise Exception(f"Failed to load model: {e2}")
        self.model = self.model.to(self.device)
        self.model.eval() # Inference mode
        
        if hasattr(self.model, "trunk"):
            self.model.trunk.chunk_size = self.chunk_size

    def unload(self):
        """Unload model from GPU."""
        print("Unloading ESMFold from GPU...")
        if self.model:
            self.model.cpu()
            del self.model
            self.model = None
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            print("ESMFold unloaded.")

    def fold(self, sequences: List[ProteinSequence], **kwargs) -> List[StructurePrediction]:
        predictions = []
        
        for seq in sequences:
            sequence_str = seq.sequence
            
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            
            with torch.no_grad():
                tokenized_input = self.tokenizer([sequence_str], return_tensors="pt", add_special_tokens=False)['input_ids']
                
                if hasattr(self.model, "hf_device_map"):
                    tokenized_input = tokenized_input.to("cuda")
                else:
                    tokenized_input = tokenized_input.to(self.device)
                
                output = self.model(tokenized_input)
                
            plddt_tensor = output.plddt[0].cpu().numpy() # Shape: (residues,)
            
            if np.max(plddt_tensor) <= 1.0:
                 plddt_tensor = plddt_tensor * 100.0
            
            pdb_string = self.convert_outputs_to_pdb(output)
            
            output_dir = kwargs.get("output_dir", "./output/pdbs")
            os.makedirs(output_dir, exist_ok=True)
            pdb_path = os.path.join(output_dir, f"{seq.id}.pdb")
            
            with open(pdb_path, "w") as f:
                f.write(pdb_string)
            
            predictions.append(StructurePrediction(
                sequence_id=seq.id,
                pdb_path=pdb_path,
                plddt=plddt_tensor,
                mean_plddt=np.mean(plddt_tensor),
                metadata={"folder": "esmfold", "model": self.model_name}
            ))
            
        return predictions

    def convert_outputs_to_pdb(self, output) -> str:
        return self.model.output_to_pdb(output)[0]

