
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
import numpy as np
from typing import List, Dict,
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import os
from datetime import datetime

# Create log directory
log_dir = os.path.join("aw_grpo_results", datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)



class RewardRegistry:
    """Registry for reward functions to make them easily configurable."""
    _rewards = {}
    _reward_history = {} 
    _slope_history = {}   
    _current_weights = {} 
    _step = 0  
    _writer = None 
    
    _weight_adjustment_rate = 0.1  
    _min_weight = 0.1  
    _max_weight = 0.8  
    _history_window = 12 
    
    @classmethod
    def set_writer(cls, writer):
        """Set TensorBoard writer"""
        cls._writer = writer
    
    @classmethod
    def _log_to_tensorboard(cls, tag, value):
        """Log a scalar to TensorBoard"""
        if cls._writer is not None:
            cls._writer.add_scalar(tag, value, cls._step)
    
    @classmethod
    def _calculate_slope(cls, history):
        """Compute slope from a reward history using linear fit"""
        if len(history) < 2:
            return 0.0
        x = np.arange(len(history))
        slope, _ = np.polyfit(x, history, 1)
        return slope
    
    @classmethod
    def update_weights(cls, reward_name: str, current_reward: float):
        """Update reward history and record slope/weights for the given reward"""
        if reward_name not in cls._reward_history:
            cls._reward_history[reward_name] = deque(maxlen=cls._history_window)
            cls._slope_history[reward_name] = deque(maxlen=cls._history_window)
            cls._current_weights[reward_name] = 1.0 
        
        # Update reward history
        cls._reward_history[reward_name].append(current_reward)
        
        # Compute slope
        slope = cls._calculate_slope(list(cls._reward_history[reward_name]))
        cls._slope_history[reward_name].append(slope)
        
        # Log to TensorBoard
        cls._log_to_tensorboard(f"rewards/{reward_name}", current_reward)
        cls._log_to_tensorboard(f"slopes/{reward_name}", slope)
        cls._log_to_tensorboard(f"weights/{reward_name}", cls._current_weights[reward_name])
    

    @classmethod
    def adjust_weights(cls):
        """Adjust weights via an exponentiated gradient-style update"""
        
        # Collect the latest slopes
        slopes = {
            name: cls._slope_history[name][-1] if cls._slope_history[name] else 0.0
            for name in cls._reward_history.keys()
        }
        if not slopes:
            print("No reward functions registered; skipping.")
            return
        
        updated = {
            name: cls._current_weights.get(name, 1.0) * np.exp(-1 * s)
            for name, s in slopes.items()
        }

        clipped = {
            name: np.clip(w, cls._min_weight, cls._max_weight)
            for name, w in updated.items()
        }
    
        total = sum(clipped.values())
        cls._current_weights = {
            name: w / total
            for name, w in clipped.items()
        }
        
        for name in slopes:
            cls._log_to_tensorboard(f"weights/{name}", cls._current_weights[name])
            cls._log_to_tensorboard(f"slopes/{name}", slopes[name])
        
        cls._step += 1
    
    @classmethod
    def get_weight(cls, reward_name: str) -> float:
        """Get the current weight for the given reward name"""
        return cls._current_weights.get(reward_name, 1.0)
    
    @classmethod
    def register(cls, name):
        """Decorator to register a reward function."""
        def decorator(func):
            cls._rewards[name] = func
            return func
        return decorator
    
    @classmethod
    def get(cls, name):
        """Get a reward function by name."""
        if name not in cls._rewards:
            raise ValueError(f"Reward function '{name}' not found")
        return cls._rewards[name]
    
    @classmethod
    def list_rewards(cls):
        """List all registered reward functions."""
        return list(cls._rewards.keys())



@RewardRegistry.register("readability")
def readability_reward(completions: List[str], ground_truth: List[str] = None, **kwargs) -> List[float]:
    try:
        from jreadability import compute_readability
        from fugashi import Tagger
    except ImportError:
        print("Warning: jreadability or fugashi not installed. Using dummy readability scores.")
        return [0.5] * len(completions)
    
    # Initialize the tagger once for batch processing
    tagger = Tagger()
    
    scores = []
    for translation in completions:
        try:
            readability_score = compute_readability(translation, tagger)
            normalized_score = min(1.0, max(0.0, readability_score / 7.0))
            scores.append(normalized_score)
        except Exception as e:
            print(f"Error calculating readability: {e}")
            scores.append(0.5)  # Neutral score
    
    return scores


@RewardRegistry.register("bleurt")
def bleurt_reward(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    model_name = kwargs.get('bleurt_model', "lucadiliello/BLEURT-20-D12")
    
    config = BleurtConfig.from_pretrained(model_name)
    model = BleurtForSequenceClassification.from_pretrained(model_name)
    tokenizer = BleurtTokenizer.from_pretrained(model_name)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    scores = []
    
    batch_size = 8
    for i in range(0, len(completions), batch_size):
        batch_completions = completions[i:i+batch_size]
        batch_references = ground_truth[i:i+batch_size]
        
        inputs = tokenizer(
            batch_references,
            batch_completions,
            padding=True,
            truncation=True,
            max_length=512,  
            return_tensors="pt"
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            batch_scores = outputs.logits.squeeze(-1).tolist()
            
        if not isinstance(batch_scores, list):
            batch_scores = [batch_scores]
            
        normalized_scores = [(score + 1) / 2 for score in batch_scores]
        scores.extend(normalized_scores)
    
    return scores



@RewardRegistry.register("combined")
def combined_reward(completions: List[str], ground_truth: List[str], **kwargs) -> List[float]:
    readability_scores = readability_reward(completions, ground_truth, **kwargs)
    bleurt_scores = bleurt_reward(completions, ground_truth, **kwargs)
    
    avg_readability = sum(readability_scores) / len(readability_scores)
    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)
    
    RewardRegistry.update_weights("readability", avg_readability)
    RewardRegistry.update_weights("bleurt", avg_bleurt)
    
    # Adjust weights
    RewardRegistry.adjust_weights()
    
    readability_weight = RewardRegistry.get_weight("readability")
    bleurt_weight = RewardRegistry.get_weight("bleurt")
    
    # Combine rewards
    combined_scores = []
    for i in range(len(completions)):
        combined_score = (
            readability_weight * readability_scores[i] +
            bleurt_weight * bleurt_scores[i]
        )
        combined_scores.append(combined_score)
    
    # Log average combined reward to TensorBoard
    avg_combined = sum(combined_scores) / len(combined_scores)
    if RewardRegistry._writer is not None:
        RewardRegistry._writer.add_scalar("rewards/combined", avg_combined, RewardRegistry._step)
    
    return combined_scores



def parse_reward_weights(weights_str: str) -> Dict[str, float]:
    weights = {}
    for item in weights_str.split(','):
        if '=' in item:
            key, value = item.split('=')
            weights[key.strip()] = float(value.strip())
    

    required_weights = ['readability', 'bleurt']
    if not all(k in weights for k in required_weights):
        weights = {
            'readability': 0.5,
            'bleurt': 0.5
        }
    
    return weights