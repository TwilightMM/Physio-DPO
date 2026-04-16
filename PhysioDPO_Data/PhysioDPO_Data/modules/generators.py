import os
import torch
import uuid
import random
from typing import List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from huggingface_hub import snapshot_download
from core.interfaces import BaseGenerator
from core.data_models import ProteinSequence

hf_mirrors = [
    'https://hf-mirror.com',
    'https://huggingface.co',
]

# Configure timeout and retry behavior
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '30'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'


hf_endpoint = os.environ.get('HF_ENDPOINT')
if not hf_endpoint:
    hf_endpoint = hf_mirrors[0]
    os.environ['HF_ENDPOINT'] = hf_endpoint

os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(os.getcwd(), 'physio_dpo', 'models')
print(f"Using HuggingFace endpoint: {hf_endpoint}")
print(f"Download timeout set to: 300 seconds")

class HuggingFaceGenerator(BaseGenerator):
    """
    Generates sequences using a HuggingFace Protein LM (e.g., nferuz/ProtGPT2).
    """
    def __init__(self, model_name: str = "nferruz/ProtGPT2", device: Optional[str] = None, 
                 min_length: int = 50, max_length: int = 200):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.min_length = min_length
        self.max_length = max_length
        
        print(f"Loading Generator model: {model_name} on {self.device}...")
        print("Downloading tokenizer and model files (this may take several minutes for first download)...")
        
        # Ensure the cache directory exists
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
            print(f"Error: {e}")
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
                raise Exception(f"Failed to load tokenizer from both mirror and original site: {e2}")
        
        # Optimization: Use float16 on CUDA
        dtype = torch.float16 if "cuda" in str(self.device) else torch.float32
        
        # Distributed Strategy: Check for multi-gpu availability
        device_map = None
        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"Detected {torch.cuda.device_count()} GPUs. Enabling distributed inference (device_map='auto').")
            device_map = "auto"
        
        print("Downloading model files (this may take a while, please wait)...")
        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                cache_dir=cache_dir,
                dtype=dtype,
                device_map=device_map,
                local_files_only=False
            )
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Failed to load model from mirror, trying original HuggingFace site...")
            print(f"Error: {e}")
            original_endpoint = os.environ.get('HF_ENDPOINT')
            os.environ['HF_ENDPOINT'] = 'https://huggingface.co'
            try:
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, 
                    cache_dir=cache_dir,
                    dtype=dtype,
                    device_map=device_map,
                    local_files_only=False
                )
                print("Model loaded from original site!")
            except Exception as e2:
                if original_endpoint:
                    os.environ['HF_ENDPOINT'] = original_endpoint
                raise Exception(f"Failed to load model from both mirror and original site: {e2}")
        
        if not device_map:
            self.model = self.model.to(self.device)
        
        # Create pipeline for easier generation
        # Note: ProtGPT2 is a CausalLM
        pipeline_kwargs = {
            "task": "text-generation",
            "model": self.model,
            "tokenizer": self.tokenizer,
            "dtype": dtype
        }
        
        if not device_map:
            pipeline_kwargs["device"] = 0 if "cuda" in str(self.device) else -1
            
        self.generator = pipeline(**pipeline_kwargs)

    def generate(self, n_samples: int, **kwargs) -> List[ProteinSequence]:
        sequences = []
        
        # Generate in batches or loops
        # ProtGPT2 generates sequences with newlines; we need to clean them.
        
        print(f"Generating {n_samples} sequences...")
        
        # We generate one by one or in small batches to control output better
        for _ in range(n_samples):
            # Start with a standard start token if required, or empty for unconditional
            # ProtGPT2 often works well with unconditional generation or a start token "<|endoftext|>"
            
            output = self.generator(
                "<|endoftext|>", 
                max_length=self.max_length, 
                min_length=self.min_length,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=1.0,
                num_return_sequences=1,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
            raw_text = output[0]['generated_text']
            
            # Clean up: Remove special tokens and newlines
            clean_seq = raw_text.replace("<|endoftext|>", "").replace("\n", "").strip()
            
            # Basic validation: ensure it looks like a protein
            if not clean_seq:
                continue
                
            seq_id = str(uuid.uuid4())
            sequences.append(ProteinSequence(
                id=seq_id, 
                sequence=clean_seq, 
                metadata={"source": self.model_name, "generation_params": kwargs}
            ))
            
        return sequences

    def unload(self):
        """Unload model from GPU to free up memory."""
        print("Unloading Generator from GPU...")
        if self.model:
            self.model.cpu()
            del self.model
            del self.generator
            self.model = None
            self.generator = None
            if "cuda" in str(self.device):
                torch.cuda.empty_cache()
            print("Generator unloaded.")

class MutationGenerator(BaseGenerator):
    """Generates sequences by mutating a list of seed sequences."""
    
    def __init__(self, seed_sequences: List[ProteinSequence], mutations_per_seq: int = 1, mutation_rate: float = 0.05):
        self.seed_sequences = seed_sequences
        self.mutations_per_seq = mutations_per_seq
        self.mutation_rate = mutation_rate
        self.vocab = "ACDEFGHIKLMNPQRSTVWY"

    def generate(self, n_samples: int = 0, **kwargs) -> List[ProteinSequence]:
        """
        Generates mutants. n_samples is ignored/inferred from seed sequences * mutations_per_seq.
        """
        generated = []
        for seed in self.seed_sequences:
            for i in range(self.mutations_per_seq):
                mutated_seq_list = list(seed.sequence)
                n_mutations = max(1, int(len(seed.sequence) * self.mutation_rate))
                
                # Perform random substitutions
                indices = random.sample(range(len(seed.sequence)), n_mutations)
                for idx in indices:
                    mutated_seq_list[idx] = random.choice(self.vocab)
                
                mutated_seq_str = "".join(mutated_seq_list)
                new_id = f"{seed.id}_mut_{i}"
                
                generated.append(ProteinSequence(
                    id=new_id,
                    sequence=mutated_seq_str,
                    metadata={
                        "source": "mutation",
                        "parent_id": seed.id,
                        "mutations_count": n_mutations
                    }
                ))
        return generated
