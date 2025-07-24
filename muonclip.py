import torch
import math
import torch.nn as nn

from transformers.models.gptj.modeling_gptj import GPTJAttention

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
    list of param_groups with the `use_muon` flag set.

    The point of this class is to allow the user to have a single optimizer in their code, rather
    than having both a MuonClip and an Adam which each need to be stepped.
    """
    def __init__(self, param_groups, model=None):
        for group in param_groups:
            assert "use_muon" in group
            if group["use_muon"]:
                # defaults for MuonClip
                group["lr"] = group.get("lr", 0.02)
                group["momentum"] = group.get("momentum", 0.95)
                group["weight_decay"] = group.get("weight_decay", 0)
                group["qk_clip_threshold"] = group.get("qk_clip_threshold", 100.0)
                group["ns_steps"] = group.get("ns_steps", 5)
                group["nesterov"] = group.get("nesterov", True)
                assert set(group.keys()) == set(["params", "lr", "momentum", "weight_decay", "qk_clip_threshold", "ns_steps", "nesterov", "use_muon"])
            else:
                # defaults for Adam
                group["lr"] = group.get("lr", 3e-4)
                group["betas"] = group.get("betas", (0.9, 0.95))
                group["eps"] = group.get("eps", 1e-10)
                group["weight_decay"] = group.get("weight_decay", 0)
                assert set(group.keys()) == set(["params", "lr", "betas", "eps", "weight_decay", "use_muon"])
        super().__init__(param_groups, dict())

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group["use_muon"]:
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
            if group.get("use_muon", False):
                qk_threshold = group['qk_clip_threshold']
                break
            
        if qk_threshold is None:
            return
        
        def apply(param, max_logit, qk_threshold, apply_sqrt=True):
            if max_logit > qk_threshold:
                gamma = qk_threshold / max_logit
                sqrt_gamma = math.sqrt(gamma)
                param.mul_(sqrt_gamma if apply_sqrt else gamma)
            
            

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


class _GPTJAttentionWithQKClip(GPTJAttention):
    """
    GPT-J attention with Muon support.
    """
    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
    ):
        # Keep the attention weights computation in fp32 to avoid overflow issues
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        attn_weights = torch.matmul(query, key.transpose(-1, -2))
        attn_weights = attn_weights / self.scale_attn

        with torch.no_grad():
            max_logits_per_head = attn_weights.amax(dim=(0, 2, 3))  # shape: [num_heads]
            
            if not hasattr(self, '_qk_clip_max_logits'):
                self._qk_clip_max_logits = max_logits_per_head
            else:
                self._qk_clip_max_logits = torch.maximum(self._qk_clip_max_logits, max_logits_per_head)

        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key.shape[-2]]
            attn_weights = attn_weights + causal_mask

        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.to(value.dtype)
        attn_weights = self.attn_dropout(attn_weights)

        # Mask heads if we want to
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights

