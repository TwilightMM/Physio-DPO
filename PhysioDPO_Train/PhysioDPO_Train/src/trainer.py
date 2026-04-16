import torch
import torch.nn.functional as F
from trl import DPOTrainer
from typing import Dict, List, Union, Tuple, Any, Optional
import inspect
import re
import time
import logging
from tqdm import tqdm

class PhysioDPOTrainer(DPOTrainer):
    def __init__(self, energy_params: Optional[Dict[str, float]] = None, *args, **kwargs):
        """
        energy_params: {'mu': 50.0, 'tau': 10.0, 'lambda': 1.0}
        """
        # Extract energy params before calling super().__init__
        if energy_params is None:
            energy_params = {}
        self.mu = energy_params.get('mu', 50.0)
        self.tau = energy_params.get('tau', 10.0)
        self.lam = energy_params.get('lambda', 1.0)
        
        # Initialize training metrics tracking
        self.current_training_step = 0
        self.total_steps = 0
        self.start_time = None
        self.train_losses = []
        self.train_metrics_history = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        try:
            sig = inspect.signature(DPOTrainer.__init__)
            accepted = set(sig.parameters.keys())
            accepted.discard("self")
            filtered_kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        except Exception:
            filtered_kwargs = dict(kwargs)

        while True:
            try:
                super().__init__(*args, **filtered_kwargs)
                break
            except TypeError as e:
                msg = str(e)
                m = re.search(r"got an unexpected keyword argument '([^']+)'", msg)
                if not m:
                    raise
                bad_key = m.group(1)
                if bad_key not in filtered_kwargs:
                    raise
                filtered_kwargs.pop(bad_key)
        
        # Store total steps for progress tracking
        self.total_steps = getattr(self.args, 'max_steps', 0)
        print(f"\n=== PhysioDPO Training Configuration ===")
        print(f"Total training steps: {self.total_steps}")
        print(f"Physio parameters - mu: {self.mu}, tau: {self.tau}, lambda: {self.lam}")
        print(f"Logging frequency: every {getattr(self.args, 'logging_steps', 10)} steps")
        print("========================================\n")

    def physio_weighting(self, energy_gap: torch.Tensor) -> torch.Tensor:
        """
        Compute physical weighting coefficient Psi(delta_E)
        Formula: lambda * sigmoid((delta_E - mu) / tau)
        """
        # Ensure numerical stability to prevent overflow
        scaled_gap = (energy_gap - self.mu) / self.tau
        weights = self.lam * torch.sigmoid(scaled_gap)
        return weights

    def concatenated_forward(self, model, batch, **kwargs):
        """
        Override to ensure input tensors have correct dtype before forward pass.
        """
        # Cast tensors to the expected dtype: ids/labels/masks must be integer tensors.
        # Fix dtypes in batch before forward
        for key in list(batch.keys()):
            if isinstance(batch[key], torch.Tensor):
                if any(k in key for k in ['input_ids', 'labels', 'concatenated_input_ids']):
                    if batch[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        batch[key] = batch[key].long()
                elif 'attention_mask' in key:
                    if batch[key].dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        batch[key] = batch[key].long()
        
        return super().concatenated_forward(model, batch, **kwargs)

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_free: bool = False,
        *args,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Delegate to TRL's built-in DPO loss.

        Physio weighting (if desired) is applied in get_batch_loss_metrics based on
        an optional `energy_gap` field in the batch.
        """
        try:
            return super().dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                reference_free=reference_free,
            )
        except TypeError:
            return super().dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
            )

    def _fix_input_dtypes(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Ensure input_ids and related tensors have correct dtype (Long/Int) for embedding layer.
        This fixes the 'Expected Long but got FloatTensor' error.
        """
        fixed_inputs = {}
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                # input_ids and labels must be Long type for embedding lookup
                if any(k in key for k in ['input_ids', 'labels', 'attention_mask']):
                    if value.dtype in [torch.float32, torch.float16, torch.bfloat16]:
                        fixed_inputs[key] = value.long()
                    else:
                        fixed_inputs[key] = value
                else:
                    fixed_inputs[key] = value
            else:
                fixed_inputs[key] = value
        return fixed_inputs

    def get_batch_loss_metrics(
        self,
        model,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        train_eval: str = "train",
    ):
        # Fix input dtypes before processing
        inputs = self._fix_input_dtypes(inputs)
        
        energy_gaps = inputs.get("energy_gap", None)

        loss, metrics = super().get_batch_loss_metrics(model, inputs, train_eval=train_eval)

        if energy_gaps is not None and not isinstance(energy_gaps, dict):
            prefix = "eval_" if train_eval == "eval" else ""
            try:
                energy_gaps = torch.as_tensor(energy_gaps, dtype=torch.float32, device=loss.device)
            except Exception:
                return loss, metrics
            weights = self.physio_weighting(energy_gaps)
            avg_w = weights.mean()

            # Reweight the scalar loss (best-effort, avoids breaking TRL internals)
            loss = loss * avg_w

            metrics[f"{prefix}physio/avg_energy_gap"] = energy_gaps.mean().detach().cpu()
            metrics[f"{prefix}physio/avg_weight"] = avg_w.detach().cpu()

        # Track training metrics if in training mode
        if train_eval == "train":
            self.current_training_step += 1
            self.train_losses.append(loss.item() if hasattr(loss, 'item') else float(loss))
            
            # Store metrics for history
            current_metrics = {k: v.item() if hasattr(v, 'item') else float(v) for k, v in metrics.items()}
            current_metrics['loss'] = self.train_losses[-1]
            current_metrics['step'] = self.current_training_step
            self.train_metrics_history.append(current_metrics)
            
            # Display progress
            self._display_training_progress(loss, metrics)

        return loss, metrics
    
    def _display_training_progress(self, loss, metrics):
        """Display training progress with detailed metrics"""
        if self.start_time is None:
            self.start_time = time.time()
            
        # Calculate progress
        progress = (self.current_training_step / self.total_steps * 100) if self.total_steps > 0 else 0
        elapsed_time = time.time() - self.start_time
        
        # Estimate remaining time
        if self.current_training_step > 0 and self.total_steps > 0:
            steps_per_sec = self.current_training_step / elapsed_time
            remaining_steps = self.total_steps - self.current_training_step
            eta_seconds = remaining_steps / steps_per_sec if steps_per_sec > 0 else 0
            eta_str = f"{eta_seconds//60:.0f}m{eta_seconds%60:.0f}s"
        else:
            eta_str = "--m--s"
            
        # Format loss
        loss_val = loss.item() if hasattr(loss, 'item') else float(loss)
        
        # Build progress bar
        bar_length = 30
        filled_length = int(bar_length * self.current_training_step // self.total_steps) if self.total_steps > 0 else 0
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Format metrics for display
        metric_strs = []
        for key, value in metrics.items():
            if isinstance(value, (torch.Tensor, float)):
                val = value.item() if hasattr(value, 'item') else float(value)
                if abs(val) < 0.001 or abs(val) > 1000:
                    metric_strs.append(f"{key}: {val:.2e}")
                else:
                    metric_strs.append(f"{key}: {val:.4f}")
        
        # Display progress line
        progress_line = (
            f"Step [{self.current_training_step:5d}/{self.total_steps:5d}] "
            f"[{progress:5.1f}%] [{bar}] "
            f"Loss: {loss_val:.6f} "
            f"ETA: {eta_str} "
        )
        
        # Add metrics if available
        if metric_strs:
            progress_line += "| " + " ".join(metric_strs[:3])  # Show first 3 metrics
            if len(metric_strs) > 3:
                progress_line += "..."
        
        print(f"\r{progress_line}", end='', flush=True)
        
        # New line every logging_steps or at the end
        logging_steps = getattr(self.args, 'logging_steps', 100)
        if self.current_training_step % logging_steps == 0 or self.current_training_step >= self.total_steps:
            print()  # New line
            
            # Log detailed metrics
            self.logger.info(f"Training Step {self.current_training_step}/{self.total_steps} ({progress:.1f}%)")
            self.logger.info(f"  Loss: {loss_val:.6f}")
            self.logger.info(f"  Learning Rate: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            for key, value in metrics.items():
                if isinstance(value, (torch.Tensor, float)):
                    val = value.item() if hasattr(value, 'item') else float(value)
                    self.logger.info(f"  {key}: {val:.6f}")
            
            # Calculate and log training speed
            if elapsed_time > 0:
                speed = self.current_training_step / elapsed_time
                self.logger.info(f"  Speed: {speed:.2f} steps/sec")
                self.logger.info(f"  Elapsed: {elapsed_time//60:.0f}m{elapsed_time%60:.0f}s")
    
    def on_train_end(self, *args, **kwargs):
        """Called when training ends"""
        print("\n")
        print("=== Training Complete ===")
        
        if self.train_losses:
            final_loss = self.train_losses[-1]
            avg_loss = sum(self.train_losses) / len(self.train_losses)
            min_loss = min(self.train_losses)
            
            print(f"Final Loss: {final_loss:.6f}")
            print(f"Average Loss: {avg_loss:.6f}")
            print(f"Best Loss: {min_loss:.6f}")
            
            if len(self.train_losses) > 1:
                # Calculate loss trend (last 10% vs first 10%)
                n_trend = max(1, len(self.train_losses) // 10)
                early_avg = sum(self.train_losses[:n_trend]) / n_trend
                late_avg = sum(self.train_losses[-n_trend:]) / n_trend
                improvement = (early_avg - late_avg) / early_avg * 100
                print(f"Loss Improvement: {improvement:.2f}%")
        
        if self.start_time:
            total_time = time.time() - self.start_time
            print(f"Total Training Time: {total_time//60:.0f}m{total_time%60:.0f}s")
            
            if self.current_training_step > 0:
                avg_speed = self.current_training_step / total_time
                print(f"Average Speed: {avg_speed:.2f} steps/sec")
        
        print("========================")
        
        # Call parent's on_train_end if it exists
        if hasattr(super(), 'on_train_end'):
            super().on_train_end(*args, **kwargs)