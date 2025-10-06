''''
from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch

from .experience_maker_ssr import Experience_ssr

@dataclass
class SSRGroup:
    experiences: List[Experience_ssr]
    rewards: List[torch.Tensor]  # Store individual reward tensors per sample

class SSRBuffer:
    def __init__(self, max_group_buffer_size: int = 1000, min_variance: float = 1e-4):
        self.group_buffer: List[SSRGroup] = []
        self.max_group_buffer_size = max_group_buffer_size
        self.min_variance = min_variance

    def append_group(self, experiences: List[Experience_ssr], rewards: List[torch.Tensor]):
        reward_group = torch.stack(rewards).reshape(-1).to(dtype=torch.float32)
        if reward_group.std().item() < self.min_variance:
            return
        if len(self.group_buffer) >= self.max_group_buffer_size:
            self.group_buffer.pop(0)
        self.group_buffer.append(
            SSRGroup(
                experiences=[e.to_device("cpu") for e in experiences],
                rewards=[r.detach().cpu() for r in rewards]
            )
        )

    def ssr_fallback(self) -> Optional[Tuple[List[Experience_ssr], List[torch.Tensor]]]:
        if not self.group_buffer:
            print("[SSR-BUFFER] ssr_fallback called but buffer is empty.")
            return None
        idx = torch.argmax(torch.tensor([
            torch.stack(g.rewards).reshape(-1).std().item() for g in self.group_buffer
        ])).item()
        stds = [torch.stack(g.rewards).reshape(-1).std().item() for g in self.group_buffer]
        print(f"[SSR-BUFFER] SSR fallback activated. Selected group #{idx} with std = {stds[idx]:.6f}")
        group = self.group_buffer[idx]
        return group.experiences, group.rewards

    def __len__(self):
        print(f"[SSR-BUFFER] Buffer length queried: {len(self.group_buffer)}")
        return len(self.group_buffer)


# SSRManager remains in experience_maker.py and interacts with SSRBuffer

class SSRManager:
    def __init__(self, buffer: SSRBuffer, min_variance=1e-5, fallback_prob=0.5):
        self.buffer = buffer
        self.min_variance = min_variance
        self.fallback_prob = fallback_prob

    def should_fallback(self, reward_group: torch.Tensor) -> bool:
        if reward_group.ndim > 1:
            reward_group = reward_group.mean(dim=-1)
        var = reward_group.var().item()
        return var < self.min_variance

    def maybe_sample(self, reward_group: torch.Tensor, fallback_fn):
        if self.should_fallback(reward_group):
            print("[SSR] Advantage variance too small. Triggering fallback...")
            if len(self.buffer) > 0:
                return self.buffer.ssr_fallback()
            else:
                print("[SSR] Buffer empty. Skipping fallback.")
        return fallback_fn()  # Return (experiences, rewards)

    def try_append(self, experiences: List[Experience_ssr], rewards: List[torch.Tensor]):
        reward_group = torch.stack(rewards).reshape(-1).to(dtype=torch.float32)
        if not self.should_fallback(reward_group):
            self.buffer.append_group(experiences, rewards)
''''



from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import random
from copy import deepcopy


@dataclass
class SSRGroup:
    experiences: List  # Experience_ssr objects
    raw_rewards: List[torch.Tensor]  # Raw rewards before compute_reward
    group_variance: float
    
    
class SSRBuffer:
    def __init__(self, max_buffer_size: int = 100, min_variance: float = 1e-4):
        self.buffer: List[SSRGroup] = []
        self.max_buffer_size = max_buffer_size
        self.min_variance = min_variance
        
    def append_group(self, experiences: List, rewards: List[torch.Tensor]):
        """Store experiences with their raw rewards"""
        # Calculate group-level reward variance
        reward_flat = torch.cat([r.flatten() for r in rewards])
        group_var = reward_flat.var().item()
        
        if group_var < self.min_variance:
            print(f"[SSR-BUFFER] Skipping low variance group: {group_var:.6f}")
            return
            
        # Remove oldest if buffer full
        if len(self.buffer) >= self.max_buffer_size:
            self.buffer.pop(0)
            
        # Store on CPU to save memory
        ssr_group = SSRGroup(
            experiences=[deepcopy(exp).to_device("cpu") for exp in experiences],
            raw_rewards=[r.detach().cpu().clone() for r in rewards],
            group_variance=group_var
        )
        
        self.buffer.append(ssr_group)
        print(f"[SSR-BUFFER] Added group with variance {group_var:.6f}, buffer size: {len(self.buffer)}")
        
    def sample_high_variance_group(self) -> Optional[SSRGroup]:
        """Sample a group with high variance from buffer"""
        if not self.buffer:
            return None
            
        # Select from top 50% highest variance groups
        sorted_groups = sorted(self.buffer, key=lambda x: x.group_variance, reverse=True)
        top_half = sorted_groups[:max(1, len(sorted_groups) // 2)]
        
        selected = random.choice(top_half)
        print(f"[SSR-BUFFER] Selected group with variance {selected.group_variance:.6f}")
        
        return selected
        
    def get_buffer_stats(self):
        if not self.buffer:
            return {"size": 0, "avg_variance": 0, "max_variance": 0}
        
        variances = [g.group_variance for g in self.buffer]
        return {
            "size": len(self.buffer),
            "avg_variance": sum(variances) / len(variances),
            "max_variance": max(variances)
        }


class SSRManager:
    def __init__(self, buffer: SSRBuffer, 
                 variance_threshold: float = 1e-4,
                 reward_percentile_threshold: float = 0.7,
                 min_buffer_size: int = 5):
        self.buffer = buffer
        self.variance_threshold = variance_threshold
        self.reward_percentile_threshold = reward_percentile_threshold
        self.min_buffer_size = min_buffer_size
        
        # 统计信息用于动态阈值
        self.reward_history = []
        self.variance_history = []
        self.update_count = 0
        
    def should_store_batch(self, raw_rewards: List[torch.Tensor]) -> bool:
        """智能判断是否应该存储当前batch"""
        reward_flat = torch.cat([r.flatten() for r in raw_rewards])
        variance = reward_flat.var().item()
        mean_reward = reward_flat.mean().item()
        
        # 更新统计历史
        self.reward_history.append(mean_reward)
        self.variance_history.append(variance)
        
        # 保持历史窗口大小
        if len(self.reward_history) > 100:
            self.reward_history.pop(0)
            self.variance_history.pop(0)
        
        # 条件1: Buffer还不够，需要填充
        #if len(self.buffer.buffer) < self.min_buffer_size:
        #    print(f"[SSR-UPDATE] Storing due to insufficient buffer size")
        #    return True
        
        # 条件2: 高方差batch（多样性好）
        if variance > self.variance_threshold * 3:
            print(f"[SSR-UPDATE] Storing high variance batch: {variance:.6f}")
            return True
        
        # 条件3: 高奖励batch（质量好）
        #if len(self.reward_history) > 10:
        #    reward_90th = torch.tensor(self.reward_history).quantile(self.reward_percentile_threshold)
        #    if mean_reward > reward_90th:
        #        print(f"[SSR-UPDATE] Storing high reward batch: {mean_reward:.3f} > {reward_90th:.3f}")
        #        return True
        
        # 条件4: 定期强制更新（防止buffer过期）
        if self.update_count % 50 == 0:  # 每50个batch强制更新一次
            print(f"[SSR-UPDATE] Periodic forced update")
            return True
        
        return False
    
    def update_buffer_if_needed(self, experiences: List, raw_rewards: List[torch.Tensor]):
        """根据策略更新buffer"""
        self.update_count += 1
        
        if self.should_store_batch(raw_rewards):
            self.buffer.append_group(experiences, raw_rewards)
            
            # 打印buffer状态
            if self.update_count % 10 == 0:
                stats = self.buffer.get_buffer_stats()
                print(f"[SSR-BUFFER] Size: {stats['size']}, "
                      f"Avg Variance: {stats['avg_variance']:.6f}, "
                      f"Max Variance: {stats['max_variance']:.6f}")

