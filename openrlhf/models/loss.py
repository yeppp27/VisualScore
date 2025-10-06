from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .utils import masked_mean


class GPTLMLoss(nn.Module):
    """
    GPT Language Model Loss
    """

    def __init__(self, ring_attn_group=None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)

        self.ring_attn_group = ring_attn_group
        if self.ring_attn_group:
            self.ring_attn_rank = dist.get_rank(self.ring_attn_group)
            self.ring_attn_world_size = dist.get_world_size(self.ring_attn_group)

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # RingAttention
        if self.ring_attn_group is not None:
            total_seq_len = labels.size(-1)
            seq_len_per_process = total_seq_len // self.ring_attn_world_size
            start_idx = self.ring_attn_rank * seq_len_per_process
            end_idx = min(start_idx + seq_len_per_process, total_seq_len)
            labels = labels[..., start_idx:end_idx]

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # if labels are all IGNORE_INDEX, then nn.CrossEntropyLoss will be nan
            if torch.all(shift_labels == self.IGNORE_INDEX):
                # Use mean of logits multiplied by 0 to maintain gradient flow
                loss = shift_logits.mean() * 0
            else:
                loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=self.ring_attn_group)
            loss = loss / self.ring_attn_world_size
        else:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = self.loss(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss


'''class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss'''


'''class PolicyLoss(nn.Module):
    """
    Policy Loss for PPO
    """

    def __init__(self, clip_eps: float = 0.2, use_dapo: bool = False, use_clip_higher: bool = False, clip_eps_low: float = 0.2, clip_eps_high: float=0.28) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.use_dapo = use_dapo
        self.use_clip_higher = use_clip_higher
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        c = 3.0
        ratio = (log_probs - old_log_probs).exp().clamp(max=c)
        surr1 = ratio * advantages
        if self.use_clip_higher:
            #print('self.use_clip_higher',self.use_clip_higher,'ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high)',ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high))
            surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
        else:
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2)
        if self.use_dapo:
            loss = masked_mean(loss.reshape(-1), action_mask.reshape(-1))
        else:
            loss = masked_mean(loss, action_mask, dim=-1).mean()
        return loss'''

import torch
import torch.nn as nn
from typing import Optional

def masked_mean(tensor, mask, dim=None):
    """Helper function for masked mean calculation"""
    if mask is None:
        return tensor.mean() if dim is None else tensor.mean(dim=dim)
    else:
        if dim is None:
            return (tensor * mask).sum() / mask.sum().clamp(min=1)
        else:
            return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1)

class PolicyLoss(nn.Module):
    """
    
    Policy Loss for PPO with optional RLVR High-Entropy Token Selection
    """
    
    def __init__(self, 
                 clip_eps: float = 0.2, 
                 use_dapo: bool = False, 
                 use_clip_higher: bool = False, 
                 clip_eps_low: float = 0.2, 
                 clip_eps_high: float = 0.28,
                 use_high_entropy: bool = False,
                 high_entropy_rho: float = 0.2) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.use_dapo = use_dapo
        self.use_clip_higher = use_clip_higher
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.use_high_entropy = use_high_entropy
        self.high_entropy_rho = high_entropy_rho

    def forward(self,
                log_probs: torch.Tensor,
                old_log_probs: torch.Tensor,
                advantages: torch.Tensor,
                action_mask: Optional[torch.Tensor] = None,
                # Additional parameters for RLVR high-entropy mode
                logits: Optional[torch.Tensor] = None,
                use_entropy = False,
                ) -> torch.Tensor:
        
        c = 3.0
        ratio = (log_probs - old_log_probs).exp().clamp(max=c)
        surr1 = ratio * advantages
        
        if self.use_clip_higher:
            surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages
        else:
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            
        loss = -torch.min(surr1, surr2)
        
        # Apply high-entropy token selection if enabled
        if self.use_high_entropy and use_entropy:
            if logits is None:
                raise ValueError("High-entropy mode requires logits parameter")
            
            # Calculate entropy for each token position
            # H_t^i = -sum(p * log(p)) where p is the probability distribution
            log_prob_dist = torch.log_softmax(logits, dim=-1)
            prob_dist = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(prob_dist * log_prob_dist, dim=-1)  # [batch_size, seq_len]
            
            batch_size, seq_len = entropy.shape
            
            # Calculate high-entropy mask for each sequence in the batch
            high_entropy_mask = torch.zeros_like(entropy, dtype=torch.bool)
            
            # Calculate entropy threshold tau_rho^B for each batch
            if action_mask is not None:
                num_valid_tokens = action_mask.sum(dim=-1)
            else:
                num_valid_tokens = torch.full((batch_size,), seq_len, device=entropy.device)
            
            top_k = torch.ceil(self.high_entropy_rho * num_valid_tokens).long()
             
            debug = True
            if debug:
                print(f"  high_entropy_rho: {self.high_entropy_rho}")
                print(f"  num_valid_tokens: {num_valid_tokens.tolist()}")
                print(f"  top_k: {top_k.tolist()}")
                print(f"  use_dapo:{self.use_dapo}")
            
            for b in range(batch_size):
                if action_mask is not None:
                    valid_entropy = entropy[b][action_mask[b]]
                else:
                    valid_entropy = entropy[b]
                    
                if len(valid_entropy) > 0:
                    k = min(top_k[b].item(), len(valid_entropy))
                    if k > 0:
                        threshold = torch.topk(valid_entropy, k)[0][-1]  # k-th largest value
                        high_entropy_mask[b] = entropy[b] >= threshold
            
            # Apply action mask to high entropy mask
            if action_mask is not None:
                high_entropy_mask = high_entropy_mask & action_mask.bool()
                # Create combined mask for high-entropy filtering
                combined_mask = high_entropy_mask
            else:
                combined_mask = high_entropy_mask
            
            if debug:
                selected_tokens = combined_mask.sum().item()
                total_tokens = combined_mask.numel()
                print(f"  Final selected tokens: {selected_tokens}/{total_tokens} ({100*selected_tokens/total_tokens:.1f}%)")
            
            # Apply high-entropy mask to loss
            # This corresponds to the indicator function I[H^i_t >= τ_ρ^B] in Equation (6)
            if self.use_dapo:
                loss = masked_mean(loss.reshape(-1), combined_mask.reshape(-1))
            else:
                loss = masked_mean(loss, combined_mask, dim=-1).mean()
        else:
            # Original behavior when high-entropy is not used
            if self.use_dapo:
                loss = masked_mean(loss.reshape(-1), action_mask.reshape(-1) if action_mask is not None else None)
            else:
                loss = masked_mean(loss, action_mask, dim=-1).mean()
        
        return loss

class ValueLoss(nn.Module):
    """
    Value Loss for PPO
    """

    def __init__(self, clip_eps: float = None) -> None:
        super().__init__()
        self.clip_eps = clip_eps

    def forward(
        self,
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.clip_eps is not None:
            values_clipped = old_values + (values - old_values).clamp(-self.clip_eps, self.clip_eps)
            surr1 = (values_clipped - returns) ** 2
            surr2 = (values - returns) ** 2
            loss = torch.max(surr1, surr2)
        else:
            loss = (values - returns) ** 2

        loss = masked_mean(loss, action_mask, dim=-1).mean()
        return 0.5 * loss


class PairWiseLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        if margin is not None:
            loss = -F.logsigmoid(chosen_reward - reject_reward - margin)
        else:
            loss = -F.logsigmoid(chosen_reward - reject_reward)
        return loss.mean()


class LogExpLoss(nn.Module):
    """
    Pairwise Loss for Reward Model
    Details: https://arxiv.org/abs/2204.05862
    """

    def forward(
        self, chosen_reward: torch.Tensor, reject_reward: torch.Tensor, margin: torch.Tensor = None
    ) -> torch.Tensor:
        loss = torch.log(1 + torch.exp(reject_reward - chosen_reward)).mean()
        return loss


class DPOLoss(nn.Module):
    """
    DPO Loss
    """

    def __init__(self, beta: float, label_smoothing: float = 0.0, ipo: bool = False) -> None:
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.ipo = ipo

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.ipo:
            losses = (logits - 1 / (2 * self.beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        else:
            # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )

        loss = losses.mean()
        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return loss, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L742
class VanillaKTOLoss(nn.Module):
    """
    KTO loss for even sampling
    """

    def __init__(self, beta: float) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
        rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

        chosen_logratios = policy_chosen_logps - reference_chosen_logps
        rejected_logratios = policy_rejected_logps - reference_rejected_logps

        losses = torch.cat(
            (
                1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
            ),
            0,
        ).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        return losses, chosen_rewards, rejected_rewards


# Adapted from https://github.com/ContextualAI/HALOs/blob/ca9b7e3eeea220c0944ad8095d641da33f907a7e/trainers.py#L770
class KTOLoss(nn.Module):
    """
    KTO loss for uneven sampling
    """

    def __init__(
        self, beta: float, desirable_weight: float, undesirable_weight: float, world_size: int, device: torch.device
    ) -> None:
        super().__init__()
        self.beta = beta
        self.world_size = world_size
        self.device = device
        self.desirable_weight = desirable_weight
        self.undesirable_weight = undesirable_weight

    def forward(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        policy_KL_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        reference_KL_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        KL = (policy_KL_logps - reference_KL_logps).mean().detach()
        # all_reduce sums up the KL estimates across all devices (gradient will also be scaled by world size)
        dist.all_reduce(KL, op=dist.ReduceOp.SUM)
        # take average (will also scale gradients appropriately)
        KL = (KL / self.world_size).clamp(min=0)

        if policy_chosen_logps.shape[0] != 0:
            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            chosen_losses = 1 - F.sigmoid(self.beta * (chosen_logratios - KL))
            chosen_rewards = self.beta * chosen_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            chosen_losses = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)
            chosen_rewards = torch.Tensor([]).to(policy_rejected_logps.dtype).to(self.device)

        if policy_rejected_logps.shape[0] != 0:
            rejected_logratios = policy_rejected_logps - reference_rejected_logps
            rejected_losses = 1 - F.sigmoid(self.beta * (KL - rejected_logratios))
            rejected_rewards = self.beta * rejected_logratios.detach()
        else:
            # important to cast to policy_dtype; otherwise error will occur during all_gather
            rejected_losses = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)
            rejected_rewards = torch.Tensor([]).to(policy_chosen_logps.dtype).to(self.device)

        losses = torch.cat(
            (self.desirable_weight * chosen_losses, self.undesirable_weight * rejected_losses), 0
        ).mean()
        return losses, chosen_rewards, rejected_rewards, KL


# Adapted from https://github.com/microsoft/LMOps/blob/main/minillm/finetune.py#L166
class KDLoss(nn.Module):
    """
    Language Model Knowledge Distillation Loss
    """

    def __init__(self):
        super().__init__()
        self.IGNORE_INDEX = -100

    def forward(self, logits: torch.Tensor, teacher_logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        teacher_probs = F.softmax(teacher_logits, dim=-1, dtype=torch.float32)
        inf_mask = torch.isinf(logits)
        logprobs = F.log_softmax(logits, dim=-1, dtype=torch.float32)
        prod_probs = torch.masked_fill(teacher_probs * logprobs, inf_mask, 0)
        x = torch.sum(prod_probs, dim=-1).view(-1)
        mask = (label != self.IGNORE_INDEX).int()
        distil_loss = -torch.sum(x * mask.view(-1), dim=0) / torch.sum(mask.view(-1), dim=0)

        return distil_loss


class PRMLoss(nn.Module):
    """
    Process Reward Model Loss
    """

    def __init__(self, placeholder_token_id: int, reward_token_ids: Optional[list[int]] = None):
        super().__init__()
        self.IGNORE_INDEX = -100
        self.loss = nn.CrossEntropyLoss(ignore_index=self.IGNORE_INDEX)
        self.placeholder_token_id = placeholder_token_id
        self.reward_token_ids = reward_token_ids

    def forward(self, inputs: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor, *, return_acc: bool = False):
        placeholder_mask = inputs == self.placeholder_token_id
        logits = logits[placeholder_mask]
        labels = labels[placeholder_mask]

        if labels.dtype == torch.float:
            # soft label
            assert len(self.reward_token_ids) == 2, "reward_token_ids should have 2 tokens for soft labels"
            logits = logits[..., self.reward_token_ids]
            positive_labels = labels.to(logits.dtype)
            negative_labels = 1 - positive_labels
            negative_labels[positive_labels != -100] = 1 - positive_labels[positive_labels != -100]
            labels = torch.stack([positive_labels, negative_labels], dim=-1)
        elif self.reward_token_ids is not None:
            # hard label with reward_token_ids set. (otherwise the whole vocab will be trained together.)
            logits = logits[..., self.reward_token_ids]
            # this is slow....
            for i, token in enumerate(self.reward_token_ids):
                labels = torch.where(labels == token, i, labels)

        loss = self.loss(logits, labels)
        if not return_acc:
            return loss

        if labels.dtype == logits.dtype:
            labels = labels.argmax(dim=-1)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc
