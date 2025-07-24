from typing import Optional, Union
import torch
import torch.nn as nn
import json
import wandb
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPTJConfig, GPTJForCausalLM
import transformers.models.gptj.modeling_gptj
from transformers.models.gptj.modeling_gptj import GPTJAttention
from transformers.cache_utils import Cache
import random
import os
import argparse
import subprocess
import time

from muon import SingleDeviceMuonWithAuxAdam
from muonclip import MuonClipWithAuxAdam, _GPTJAttentionWithQKClip

torch.set_float32_matmul_precision('medium')

# Configuration
class Config:
    # Model
    model_name = "EleutherAI/gpt-j-6B"
    attn_impl = "eager"
    
    # Training
    batch_size = 4
    adam_lr = 1e-3
    num_steps = 2000
    log_interval = 100
    max_seq_len = 512

    # Adam hparams
    adam_beta1 = 0.8
    adam_beta2 = 0.95
    adam_eps = 1e-10

    # Muon hparams
    muon_head_lr = 0.05
    muon_embed_lr = 0.6
    muon_scalar_lr = 0.04
    muon_adam_lr = 0.22
    muon_momentum = 0.95
    muon_adam_beta1 = 0.8
    muon_adam_beta2 = 0.95
    muon_adam_eps = 1e-10

    # Muonclip hparams
    muonclip_head_lr = 0.05
    muonclip_embed_lr = 0.6
    muonclip_scalar_lr = 0.04
    muonclip_adam_lr = 0.22
    muonclip_momentum = 0.95
    muonclip_qk_clip_threshold = 100.0
    muonclip_adam_beta1 = 0.8
    muonclip_adam_beta2 = 0.95
    muonclip_adam_eps = 1e-10

    assert_same = True

    # Data
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"

config = Config()
if config.assert_same:
    assert config.muon_head_lr == config.muonclip_head_lr, "Muon and MuonClip head learning rates must match"
    assert config.muon_embed_lr == config.muonclip_embed_lr, "Muon and MuonClip embed learning rates must match"
    assert config.muon_scalar_lr == config.muonclip_scalar_lr, "Muon and MuonClip scalar learning rates must match"
    assert config.muon_adam_lr == config.muonclip_adam_lr, "Muon and MuonClip Adam learning rates must match"
    assert config.muon_momentum == config.muonclip_momentum, "Muon and MuonClip momentum must match"
    assert config.muon_adam_beta1 == config.muonclip_adam_beta1, "Muon and MuonClip Adam beta1 must match"
    assert config.muon_adam_beta2 == config.muonclip_adam_beta2, "Muon and MuonClip Adam beta2 must match"
    assert config.muon_adam_eps == config.muonclip_adam_eps, "Muon and MuonClip Adam epsilon must match"

# Set deterministic behavior
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)


# Data loading
def create_dataloader(config, tokenizer):
    # Load dataset
    dataset = load_dataset(
        config.dataset_name, 
        config.dataset_config,
        split="train",
        streaming=True,
    )
    
    # Deterministic shuffle with seed
    dataset = dataset.shuffle(seed=42, buffer_size=10000) # type: ignore
    
    def tokenize_function(examples):
        # Tokenize texts individually to respect boundaries
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for language modeling)
        tokenized["labels"] = tokenized["input_ids"].clone()
        return tokenized
    
    # Process dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names, # type: ignore
    )
    
    # Convert to torch format
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset, # type: ignore
        batch_size=config.batch_size,
        drop_last=True,
    )
    
    return dataloader

# Training function
def train_model(model, optimizer, dataloader, config, opt_name):
    wandb.init(
        project="muonclip-convergence",
        name=opt_name,
        config={
            "optimizer": opt_name,
            "config": json.dumps(config.__dict__, indent=2),
            "batch_size": config.batch_size,
            "model_size": sum(p.numel() for p in model.parameters()),
        }
    )
    
    model.train()
    step = 0
    
    # Create iterator from dataloader
    data_iter = iter(dataloader)

    # Track losses for averaging
    losses_in_interval = []

    total_samples = config.num_steps * config.batch_size
    print(f"Starting training with {opt_name}")
    print(f"Total samples to process: {total_samples:,} ({config.num_steps:,} steps × {config.batch_size} batch size)")
    start_time = time.time()
    last_log_time = start_time

    for step in range(config.num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            # Reset iterator if we run out of data
            data_iter = iter(dataloader)
            batch = next(data_iter)
        
        # Move to device
        input_ids = batch["input_ids"].to(config.device)
        labels = batch["labels"].to(config.device)
        
        # Forward pass
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step
        optimizer.step()
        
        # Track loss for averaging
        losses_in_interval.append(loss.item())
        
        # Logging
        if step % config.log_interval == 0:
            current_time = time.time()
            elapsed_time = current_time - start_time
            
            # Calculate iterations per second for this interval
            if step > 0:
                interval_time = current_time - last_log_time
                it_per_sec = config.log_interval / interval_time
            else:
                it_per_sec = 0.0
            
            # Estimate time remaining
            if step > 0:
                avg_time_per_step = elapsed_time / (step + 1)
                remaining_steps = config.num_steps - (step + 1)
                eta_seconds = remaining_steps * avg_time_per_step
                eta_str = f"{int(eta_seconds // 3600):02d}:{int((eta_seconds % 3600) // 60):02d}:{int(eta_seconds % 60):02d}"
            else:
                eta_str = "??:??:??"
            
            # Get learning rate
            lr = optimizer.param_groups[0]['lr']
            
            # Calculate average loss and perplexity for this interval
            avg_loss = np.mean(losses_in_interval)
            avg_perplexity = np.exp(avg_loss)
            
            # Log metrics
            metrics = {
                "loss": avg_loss,
                "perplexity": avg_perplexity,
                "grad_norm": grad_norm.item(),
                "learning_rate": lr,
                "step": step,
                "it_per_sec": it_per_sec,
                "elapsed_time": elapsed_time,
            }
            
            # Log attention max logits for MuonClip
            if isinstance(optimizer, MuonClipWithAuxAdam):
                max_logits = []
                for layer_id, layer_info in optimizer.attention_layers.items():
                    max_logits.extend(layer_info['max_logits'])
                if max_logits:
                    metrics["max_attention_logit"] = max(max_logits)
            
            wandb.log(metrics)
            
            # Format elapsed time
            elapsed_str = f"{int(elapsed_time // 3600):02d}:{int((elapsed_time % 3600) // 60):02d}:{int(elapsed_time % 60):02d}"
            
            print(f"Step {step}/{config.num_steps}: loss={avg_loss:.4f}, ppl={avg_perplexity:.2f}, {it_per_sec:.2f}it/s, elapsed={elapsed_str}, ETA={eta_str}")
            
            last_log_time = current_time
            # Reset losses for next interval
            losses_in_interval = []
    
    wandb.finish()
    return model

def get_param_name(model, target_param):
    for name, param in model.named_parameters():
        if id(param) == id(target_param):
            return name
    raise ValueError("Parameter not found in model")

# Main training loop
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train model with specified optimizer')
    parser.add_argument('--optimizer', type=str, required=True, 
                       choices=['adam', 'muon', 'muonclip'],
                       help='Optimizer to use for training')
    args = parser.parse_args()
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Create dataloader
    print("Loading dataset...")
    dataloader = create_dataloader(config, tokenizer)
    
    # Create model with HuggingFace GPTJ
    print("Creating model...")
    model_config = GPTJConfig.from_pretrained(config.model_name)
    model_config.use_cache = False
    model_config._attn_implementation = config.attn_impl
    model = GPTJForCausalLM(model_config).to(config.device) # type: ignore

    # Replace GPT-J attention with our custom implementation
    GPTJAttention._attn = _GPTJAttentionWithQKClip._attn # type: ignore (Monke patch :3)

    # collect the parameters to optimize    
    embed_params = [model.transformer.wte.weight]
    head_params = [model.lm_head.weight]
    accounted_names = {get_param_name(model, p) for p in (embed_params + head_params)}
    scalar_params = [p for n, p in model.named_parameters() if (p.ndim < 2) and (n not in accounted_names)]
    hidden_matrix_params = [p for n, p in model.named_parameters() if (p.ndim >= 2) and (n not in accounted_names)]

    # Create optimizer based on argument
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            list(model.parameters()), 
            lr=config.adam_lr,
            betas=(config.adam_beta1, config.adam_beta2),
            eps=config.adam_eps
        )
    elif args.optimizer == 'muon':
        adam_groups = [
            dict(params=head_params, lr=config.muon_adam_lr), 
            dict(params=embed_params, lr=config.muon_embed_lr), 
            dict(params=scalar_params, lr=config.muon_scalar_lr)
        ]
        adam_groups = [dict(**g, betas=(config.muon_adam_beta1, config.muon_adam_beta2), eps=config.muon_adam_eps, use_muon=False) for g in adam_groups]
        muon_group = dict(params=hidden_matrix_params, lr=config.muon_head_lr, momentum=config.muon_momentum, use_muon=True)
        param_groups = [*adam_groups, muon_group]
        optimizer = SingleDeviceMuonWithAuxAdam(param_groups=param_groups)
    elif args.optimizer == 'muonclip':
        adam_groups = [
            dict(params=head_params, lr=config.muonclip_adam_lr), 
            dict(params=embed_params, lr=config.muonclip_embed_lr), 
            dict(params=scalar_params, lr=config.muonclip_scalar_lr)
        ]
        adam_groups = [dict(**g, betas=(config.muonclip_adam_beta1, config.muonclip_adam_beta2), eps=config.muonclip_adam_eps, use_muon=False) for g in adam_groups]
        muon_group = dict(params=hidden_matrix_params, lr=config.muonclip_head_lr, momentum=config.muonclip_momentum, use_muon=True, qk_clip_threshold=config.muonclip_qk_clip_threshold)
        param_groups = [*adam_groups, muon_group]
        optimizer = MuonClipWithAuxAdam(param_groups=param_groups, model=model)

    opt_name = args.optimizer
    print(f"Training with {opt_name}...")
    
    # Train
    train_model(model, optimizer, dataloader, config, opt_name)

if __name__ == "__main__":
    main()
