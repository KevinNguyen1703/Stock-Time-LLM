# This source code is provided for the purposes of scientific reproducibility
# under the following limited license from Element AI Inc. The code is an
# implementation of the N-BEATS model (Oreshkin et al., N-BEATS: Neural basis
# expansion analysis for interpretable time series forecasting,
# https://arxiv.org/abs/1905.10437). The copyright to the source code is
# licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.  Any commercial use (whether
# for the benefit of third parties or internally in production) requires an
# explicit license. The subject-matter of the N-BEATS model and associated
# materials are the property of Element AI Inc. and may be subject to patent
# protection. No license to patents is granted hereunder (whether express or
# implied). Copyright © 2020 Element AI Inc. All rights reserved.

"""
Loss functions for PyTorch.
"""

import torch as t
import torch.nn as nn
import numpy as np
import pdb


def divide_no_nan(a, b):
    """
    a/b where the resulted NaN or Inf are replaced by 0.
    """
    result = a / b
    result[result != result] = .0
    result[result == np.inf] = .0
    return result


class mape_loss(nn.Module):
    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MAPE loss as defined in: https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        weights = divide_no_nan(mask, target)
        return t.mean(t.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        sMAPE loss as defined in https://robjhyndman.com/hyndsight/smape/ (Makridakis 1993)

        :param forecast: Forecast values. Shape: batch, time
        :param target: Target values. Shape: batch, time
        :param mask: 0/1 mask. Shape: batch, time
        :return: Loss value
        """
        return 200 * t.mean(divide_no_nan(t.abs(forecast - target),
                                          t.abs(forecast.data) + t.abs(target.data)) * mask)


class mase_loss(nn.Module):
    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: t.Tensor, freq: int,
                forecast: t.Tensor, target: t.Tensor, mask: t.Tensor) -> t.float:
        """
        MASE loss as defined in "Scaled Errors" https://robjhyndman.com/papers/mase.pdf

        :param insample: Insample values. Shape: batch, time_i
        :param freq: Frequency value
        :param forecast: Forecast values. Shape: batch, time_o
        :param target: Target values. Shape: batch, time_o
        :param mask: 0/1 mask. Shape: batch, time_o
        :return: Loss value
        """
        masep = t.mean(t.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        masked_masep_inv = divide_no_nan(mask, masep[:, None])
        return t.mean(t.abs(target - forecast) * masked_masep_inv)


class DirectionalLoss(nn.Module):
    """
    Combined loss: MSE + Directional penalty
    Penalizes predictions that get the direction wrong.
    
    This loss is designed for time series forecasting where the direction
    of change is as important as the magnitude.
    
    Args:
        direction_weight: Weight for the directional loss component (default: 0.3)
        use_soft_direction: Use soft directional loss (sigmoid) instead of hard (sign)
    """
    def __init__(self, direction_weight=0.3, use_soft_direction=True):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
        self.use_soft_direction = use_soft_direction
    
    def forward(self, pred, target, prev_values=None):
        """
        Compute combined MSE + Directional loss.
        
        Args:
            pred: Predicted values. Shape: (batch, pred_len, features) or (batch, pred_len, 1)
            target: Target values. Shape: (batch, pred_len, features) or (batch, pred_len, 1)
            prev_values: Previous values for direction calculation. Shape: (batch,) or (batch, features)
                         If None, direction loss is skipped.
        
        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        # Get the first prediction step for direction comparison
        # pred shape: (batch, pred_len, features) -> get (batch,)
        if len(pred.shape) == 3:
            pred_first = pred[:, 0, -1] if pred.shape[-1] > 1 else pred[:, 0, 0]
            target_first = target[:, 0, -1] if target.shape[-1] > 1 else target[:, 0, 0]
        else:
            pred_first = pred[:, 0]
            target_first = target[:, 0]
        
        # Ensure prev_values is 1D
        if len(prev_values.shape) > 1:
            prev_values = prev_values[:, -1] if prev_values.shape[-1] > 1 else prev_values[:, 0]
        
        if self.use_soft_direction:
            # Soft directional loss using sigmoid
            # Scale factor to make the sigmoid sharper
            scale = 10.0
            pred_dir_prob = t.sigmoid(scale * (pred_first - prev_values))
            target_dir_prob = t.sigmoid(scale * (target_first - prev_values))
            
            # Binary cross-entropy like loss for direction
            eps = 1e-7
            direction_loss = -t.mean(
                target_dir_prob * t.log(pred_dir_prob + eps) + 
                (1 - target_dir_prob) * t.log(1 - pred_dir_prob + eps)
            )
        else:
            # Hard directional loss
            pred_direction = t.sign(pred_first - prev_values)
            target_direction = t.sign(target_first - prev_values)
            
            # Direction mismatch penalty (not differentiable but useful for evaluation)
            direction_mismatch = (pred_direction != target_direction).float()
            direction_loss = direction_mismatch.mean()
        
        # Combined loss
        total_loss = mse_loss + self.direction_weight * direction_loss
        
        return total_loss
    
    def compute_direction_accuracy(self, pred, target, prev_values):
        """
        Compute directional accuracy (win rate) for evaluation.
        
        Returns:
            Tuple of (correct_predictions, total_predictions, accuracy_percentage)
        """
        with t.no_grad():
            if len(pred.shape) == 3:
                pred_first = pred[:, 0, -1] if pred.shape[-1] > 1 else pred[:, 0, 0]
                target_first = target[:, 0, -1] if target.shape[-1] > 1 else target[:, 0, 0]
            else:
                pred_first = pred[:, 0]
                target_first = target[:, 0]
            
            if len(prev_values.shape) > 1:
                prev_values = prev_values[:, -1] if prev_values.shape[-1] > 1 else prev_values[:, 0]
            
            pred_dir = (pred_first > prev_values).float()
            target_dir = (target_first > prev_values).float()
            
            correct = (pred_dir == target_dir).sum().item()
            total = pred.shape[0]
            accuracy = correct / total * 100 if total > 0 else 0
            
            return correct, total, accuracy


class CombinedDirectionalMSELoss(nn.Module):
    """
    Alternative implementation that combines MSE with a smoother directional penalty.
    Uses absolute difference in returns instead of binary direction.
    """
    def __init__(self, direction_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
    
    def forward(self, pred, target, prev_values=None):
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        # Compute predicted and actual returns
        if len(pred.shape) == 3:
            pred_first = pred[:, 0, -1] if pred.shape[-1] > 1 else pred[:, 0, 0]
            target_first = target[:, 0, -1] if target.shape[-1] > 1 else target[:, 0, 0]
        else:
            pred_first = pred[:, 0]
            target_first = target[:, 0]
        
        if len(prev_values.shape) > 1:
            prev_values = prev_values[:, -1] if prev_values.shape[-1] > 1 else prev_values[:, 0]
        
        # Compute returns
        pred_return = (pred_first - prev_values) / (t.abs(prev_values) + 1e-8)
        target_return = (target_first - prev_values) / (t.abs(prev_values) + 1e-8)
        
        # Penalize when signs differ, weighted by magnitude of error
        sign_match = t.sign(pred_return) * t.sign(target_return)
        direction_penalty = t.mean(t.relu(-sign_match) * t.abs(pred_return - target_return))
        
        return mse_loss + self.direction_weight * direction_penalty
