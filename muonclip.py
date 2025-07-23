import torch
import torch.nn as nn
import math

def zeropower_via_newtonschulz5(G, steps: int):
    """Newton-Schulz iteration to compute the zeroth power / orthogonalization of G."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muonclip_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """MuonClip update function"""
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp(momentum, beta) if nesterov else momentum
    
    original_shape = update.shape
    if update.ndim == 4:
        update = update.view(len(update), -1)
    
    if update.ndim >= 2:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update *= max(1, update.size(-2) / update.size(-1))**0.5
    
    if len(original_shape) == 4:
        update = update.view(original_shape)
    
    return update


def adam_update(grad, buf1, buf2, step, betas, eps):
    buf1.lerp_(grad, 1 - betas[0])
    buf2.lerp_(grad.square(), 1 - betas[1])
    buf1c = buf1 / (1 - betas[0]**step)
    buf2c = buf2 / (1 - betas[1]**step)
    return buf1c / (buf2c.sqrt() + eps)


class MuonClipWithAuxAdam(torch.optim.Optimizer):
    """
    MuonClip variant that can be used for all parameters in the network, since it runs an
    internal AdamW for the parameters that are not compatible with MuonClip. The user must manually
    specify which parameters shall be optimized with MuonClip and which with Adam by passing in a
    list of param_groups with the `use_muonclip` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a MuonClip and an Adam which each need to be stepped.
    """
    def __init__(self, param_groups, model=None):
        for group in param_groups:
            assert "use_muonclip" in group
            if group["use_muonclip"]:
                # defaults for MuonClip
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["qk_clip_threshold"] = group.get("qk_clip_threshold", 100.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "qk_clip_threshold", "ns_steps", "nesterov", "use_muonclip"])
            else:
                # defaults for Adam
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muonclip"])
        super().__init__(param_groups, dict())
        
        # Auto-detect attention layers from model structure
        self.attention_layers = {}
        if model is not None:
            self._detect_attention_params(model)

    def _detect_attention_params(self, model):
        """Automatically detect and group attention parameters from model"""
        # Walk through all modules and find attention layers
        for name, module in model.named_modules():
            if isinstance(module, (MLAAttentionWithQKClip, SimpleAttentionWithQKClip)):
                layer_id = name  # Use module path as layer ID
                
                if isinstance(module, MLAAttentionWithQKClip):
                    # MLA style with per-head components
                    heads_config = []
                    for h in range(module.num_heads):
                        heads_config.append({
                            'qc_params': [module.qc_projs[h].weight],
                            'kc_params': [module.kc_projs[h].weight],
                            'qr_params': [module.qr_projs[h].weight],
                            'kr_params': []  # kr is shared
                        })
                    
                    self.attention_layers[layer_id] = {
                        'heads': heads_config,
                        'max_logits': [0.0] * module.num_heads,
                        'module': module
                    }
                else:
                    # Standard attention
                    self.attention_layers[layer_id] = {
                        'heads': [{
                            'qc_params': [module.q_proj.weight],
                            'kc_params': [module.k_proj.weight],
                            'qr_params': [],
                            'kr_params': []
                        }],
                        'max_logits': [0.0],
                        'module': module
                    }
                
                # Set optimizer reference in module
                module._optimizer = self
                module._layer_id = layer_id

    def register_attention_layer(self, layer_id, module):
        """Manually register an attention layer for QK-Clip"""
        if isinstance(module, MLAAttentionWithQKClip):
            # MLA style with per-head components
            heads_config = []
            for h in range(module.num_heads):
                heads_config.append({
                    'qc_params': [module.qc_projs[h].weight],
                    'kc_params': [module.kc_projs[h].weight],
                    'qr_params': [module.qr_projs[h].weight],
                    'kr_params': []  # kr is shared
                })
            
            self.attention_layers[layer_id] = {
                'heads': heads_config,
                'max_logits': [0.0] * module.num_heads,
                'module': module
            }
        else:
            # Standard attention
            self.attention_layers[layer_id] = {
                'heads': [{
                    'qc_params': [module.q_proj.weight],
                    'kc_params': [module.k_proj.weight],
                    'qr_params': [],
                    'kr_params': []
                }],
                'max_logits': [0.0],
                'module': module
            }
        
        # Set optimizer reference in module
        module._optimizer = self
        module._layer_id = layer_id

    def update_attention_max_logit(self, layer_id, max_logit, head_idx=None):
        """Update max logit for a specific attention layer/head"""
        if layer_id not in self.attention_layers:
            return
            
        if isinstance(max_logit, (list, tuple)):
            self.attention_layers[layer_id]['max_logits'] = list(max_logit)
        elif head_idx is not None:
            self.attention_layers[layer_id]['max_logits'][head_idx] = max_logit
        else:
            self.attention_layers[layer_id]['max_logits'] = [max_logit] * len(
                self.attention_layers[layer_id]['heads']
            )

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muonclip"]:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["momentum_buffer"] = torch.zeros_like(p)
                    update = muonclip_update(p.grad, state["momentum_buffer"], beta=group["momentum"],
                                           ns_steps=group["ns_steps"], nesterov=group["nesterov"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])
            else:
                for p in group["params"]:
                    if p.grad is None:
                        p.grad = torch.zeros_like(p)  # Force synchronization
                    state = self.state[p]
                    if len(state) == 0:
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                        state["step"] = 0
                    state["step"] += 1
                    update = adam_update(p.grad, state["exp_avg"], state["exp_avg_sq"],
                                         state["step"], group["betas"], group["eps"])
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(update, alpha=-group["lr"])

        self._apply_qk_clip()
        
        return loss
    
    def _apply_qk_clip(self):
        """Apply QK-Clip to registered attention parameters"""
        qk_threshold = None
        for group in self.param_groups:
            if group.get("use_muonclip", False):
                qk_threshold = group['qk_clip_threshold']
                break
            
        if qk_threshold is None:
            return
        
        for layer_id, layer_info in self.attention_layers.items():
            heads = layer_info['heads']
            max_logits = layer_info['max_logits']
            
            for head_idx, (head_config, max_logit) in enumerate(zip(heads, max_logits)):
                if max_logit > qk_threshold:
                    gamma = qk_threshold / max_logit
                    sqrt_gamma = math.sqrt(gamma)
                    
                    for param in head_config.get('qc_params', []):
                        param.mul_(sqrt_gamma)
                    
                    for param in head_config.get('kc_params', []):
                        param.mul_(sqrt_gamma)
                    
                    for param in head_config.get('qr_params', []):
                        param.mul_(gamma)



# Example MLA-style attention module with QK-Clip integration
class MLAAttentionWithQKClip(nn.Module):
    """MLA-style attention module for MuonClip"""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # MLA-style decomposed projections
        self.qc_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(num_heads)
        ])
        self.kc_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(num_heads)
        ])
        self.qr_projs = nn.ModuleList([
            nn.Linear(dim, self.head_dim) for _ in range(num_heads)
        ])
        # Shared key rotary
        self.kr_proj = nn.Linear(dim, dim)
        
        self.v_proj = nn.Linear(dim, dim)
        
        # Will be set by optimizer
        self._optimizer = None
        self._layer_id = None
    
    def forward(self, x):
        B, L, D = x.shape
        
        all_max_logits = []
        outputs = []
        
        # Shared key rotary component
        kr = self.kr_proj(x)
        
        for h in range(self.num_heads):
            # Head-specific components
            qc = self.qc_projs[h](x)
            kc = self.kc_projs[h](x)
            qr = self.qr_projs[h](x)
            
            # Combine components
            q = qc + qr
            k = kc + kr[:, :, h*self.head_dim:(h+1)*self.head_dim]
            
            # Compute attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            # Track per-head max logit
            if self.training and self._optimizer is not None:
                with torch.no_grad():
                    max_logit = scores.max().item()
                    all_max_logits.append(max_logit)
            
            attn = torch.softmax(scores, dim=-1)
            outputs.append(attn)
        
        # Update optimizer with per-head max logits
        if self.training and self._optimizer is not None and all_max_logits:
            self._optimizer.update_attention_max_logit(
                self._layer_id, all_max_logits
            )
        
        # Simplified output
        avg_attn = torch.stack(outputs, dim=0).mean(dim=0)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = torch.matmul(avg_attn.unsqueeze(1), v).transpose(1, 2).contiguous()
        out = out.view(B, L, D)
        
        return out


# Standard attention module
class SimpleAttentionWithQKClip(nn.Module):
    """Standard attention module with QK-Clip support"""
    
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        
        # Will be set by optimizer
        self._optimizer = None
        self._layer_id = None
    
    def forward(self, x):
        B, L, D = x.shape
        
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if self.training and self._optimizer is not None:
            with torch.no_grad():
                max_logit = scores.max().item()
                self._optimizer.update_attention_max_logit(self._layer_id, max_logit)
        
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        
        return out

