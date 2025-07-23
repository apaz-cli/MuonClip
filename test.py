from typing import Optional, Union
import torch
import torch.nn as nn
import math
import wandb
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
from transformers.cache_utils import Cache
from tqdm import tqdm
import random
import os
import argparse

from muon import Muon
from muonclip import MuonClip, MLAAttentionWithQKClip, SimpleAttentionWithQKClip

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

# Configuration
class Config:
    # Model
    model_name = "gpt2"  # small GPT2
    
    # Training
    batch_size = 8
    learning_rate = 1e-3
    weight_decay = 0.1
    num_steps = 1000
    log_interval = 10
    max_length = 512
    
    # Optimizer specific
    adam_beta1 = 0.9
    adam_beta2 = 0.999
    muon_momentum = 0.95
    muonclip_threshold = 100.0
    
    # Data
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_config = "sample-10BT"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

config = Config()

# Adapter to make SimpleAttentionWithQKClip compatible with GPT2
class GPT2AttentionAdapter(nn.Module):
    """Adapts SimpleAttentionWithQKClip to work as a drop-in replacement for GPT2Attention"""
    
    def __init__(self, config, layer_idx=None):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        
        # Our custom attention module
        self.custom_attn = SimpleAttentionWithQKClip(self.embed_dim, self.num_heads)
        
        # Output projection (GPT2 has this separate from QKV)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)
        
        # Copy over the scale
        self.scale_attn_weights = config.scale_attn_weights
        self.layer_idx = layer_idx
        
        # Reorder for compatibility
        self.reorder_and_upcast_attn = False
    
    def forward(
        self,
        hidden_states,
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
        **kwargs,
    ) -> tuple[Union[torch.Tensor, tuple[torch.Tensor]], None]:
        # For simplicity, we ignore most of the extra features
        # and just do basic attention
        
        # Apply our custom attention
        attn_output = self.custom_attn(hidden_states)
        
        # Apply output projection and dropout
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        
        return attn_output, None

def replace_attention_modules(model):
    """Replace all GPT2Attention modules with our custom attention"""
    for name, module in model.named_children():
        if isinstance(module, GPT2Attention):
            # Get layer index from parent
            layer_idx = None
            if hasattr(model, 'layer_idx'):
                layer_idx = model.layer_idx
            
            # Replace with our adapter
            setattr(model, name, GPT2AttentionAdapter(module.config, layer_idx))
        else:
            # Recursively replace in child modules
            replace_attention_modules(module)

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
    dataset = dataset.shuffle(seed=42, buffer_size=10000)
    
    def tokenize_function(examples):
        # Tokenize texts individually to respect boundaries
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_length,
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
        remove_columns=dataset.column_names,
    )
    
    # Convert to torch format
    tokenized_dataset = tokenized_dataset.with_format("torch")
    
    # Create dataloader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=config.batch_size,
        drop_last=True,
    )
    
    return dataloader

# Training function
def train_model(model, optimizer, dataloader, config, run_name):
    wandb.init(
        project="muonclip-convergence",
        name=run_name,
        config={
            "optimizer": run_name,
            "learning_rate": config.learning_rate,
            "batch_size": config.batch_size,
            "model_size": sum(p.numel() for p in model.parameters()),
        }
    )
    
    model.train()
    step = 0
    
    # Create iterator from dataloader
    data_iter = iter(dataloader)
    
    pbar = tqdm(range(config.num_steps), desc=f"Training {run_name}")
    
    for step in pbar:
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
        
        # Logging
        if step % config.log_interval == 0:
            # Get learning rate
            lr = optimizer.param_groups[0]['lr']
            
            # Calculate perplexity
            perplexity = torch.exp(loss)
            
            # Log metrics
            metrics = {
                "loss": loss.item(),
                "perplexity": perplexity.item(),
                "grad_norm": grad_norm.item(),
                "learning_rate": lr,
                "step": step,
            }
            
            # Log attention max logits for MuonClip
            if isinstance(optimizer, MuonClip):
                max_logits = []
                for layer_id, layer_info in optimizer.attention_layers.items():
                    max_logits.extend(layer_info['max_logits'])
                if max_logits:
                    metrics["max_attention_logit"] = max(max_logits)
                    metrics["mean_attention_logit"] = np.mean(max_logits)
            
            wandb.log(metrics)
            
            pbar.set_postfix(loss=f"{loss.item():.4f}", ppl=f"{perplexity.item():.2f}")
    
    wandb.finish()
    return model

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
    
    # Create model with HuggingFace GPT2
    print("Creating model...")
    model_config = GPT2Config.from_pretrained(config.model_name)
    model_config.use_cache = False
    model = GPT2LMHeadModel(model_config)
    
    # Replace attention modules with our custom implementation
    print("Replacing attention modules...")
    replace_attention_modules(model)
    
    # Move to device
    model = model.to(config.device)
    
    # Verify replacement worked
    attention_modules = []
    for name, module in model.named_modules():
        if isinstance(module, GPT2AttentionAdapter):
            attention_modules.append(name)
    print(f"Replaced {len(attention_modules)} attention modules")
    
    # Create optimizer based on argument
    opt_name = args.optimizer.capitalize()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            betas=(config.adam_beta1, config.adam_beta2),
            weight_decay=config.weight_decay
        )
    elif args.optimizer == 'muon':
        optimizer = Muon(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay
        )
    elif args.optimizer == 'muonclip':
        optimizer = MuonClip(
            model,
            lr=config.learning_rate,
            momentum=config.muon_momentum,
            weight_decay=config.weight_decay,
            qk_clip_threshold=config.muonclip_threshold
        )
    
    print(f"Training with {opt_name}...")
    
    # Train
    trained_model = train_model(model, optimizer, dataloader, config, opt_name)
    
    # Save final loss for comparison
    model.eval()
    with torch.no_grad():
        # Get a validation batch
        val_batch = next(iter(dataloader))
        input_ids = val_batch["input_ids"].to(config.device)
        labels = val_batch["labels"].to(config.device)
        outputs = model(input_ids=input_ids, labels=labels)
        val_loss = outputs.loss
        print(f"Final validation loss: {val_loss.item():.4f}")

if __name__ == "__main__":
    main()
