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


# ============================================================================
# STOCK PREDICTION LOSS FUNCTIONS
# ============================================================================

class DirectionalLoss(nn.Module):
    """
    Directional Loss: MSE + Directional penalty
    
    Penalizes predictions that get the direction (UP/DOWN) wrong.
    Uses soft (differentiable) BCE-like loss for direction.
    
    Loss = MSE + direction_weight * DirectionBCE
    
    Args:
        direction_weight: Weight for the directional loss component (default: 0.3)
    """
    def __init__(self, direction_weight=0.3):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
    
    def forward(self, pred, target, prev_values=None):
        """
        Args:
            pred: Predicted values. Shape: (batch, pred_len, features)
            target: Target values. Shape: (batch, pred_len, features)
            prev_values: Previous values for direction calculation. Shape: (batch,)
        
        Returns:
            Combined loss value
        """
        # Standard MSE loss
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        # Get the first prediction step for direction comparison
        if len(pred.shape) == 3:
            pred_first = pred[:, 0, -1] if pred.shape[-1] > 1 else pred[:, 0, 0]
            target_first = target[:, 0, -1] if target.shape[-1] > 1 else target[:, 0, 0]
        else:
            pred_first = pred[:, 0]
            target_first = target[:, 0]
        
        # Ensure prev_values is 1D
        if len(prev_values.shape) > 1:
            prev_values = prev_values[:, -1] if prev_values.shape[-1] > 1 else prev_values[:, 0]
        
        # Soft directional loss using sigmoid + BCE
        scale = 10.0
        pred_dir_prob = t.sigmoid(scale * (pred_first - prev_values))
        target_dir_prob = t.sigmoid(scale * (target_first - prev_values))
        
        # Binary cross-entropy for direction
        eps = 1e-7
        direction_loss = -t.mean(
            target_dir_prob * t.log(pred_dir_prob + eps) + 
            (1 - target_dir_prob) * t.log(1 - pred_dir_prob + eps)
        )
        
        return mse_loss + self.direction_weight * direction_loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss: Different penalties for different direction errors.
    
    Useful when:
    - Missing UP is costly (missed profit opportunity)
    - Missing DOWN is costly (incurred loss)
    
    Loss = MSE * weight, where weight depends on error type:
    - Predicted DOWN but actual UP: weight = up_penalty (missed profit)
    - Predicted UP but actual DOWN: weight = down_penalty (incurred loss)
    - Correct direction: weight = 1.0
    
    Args:
        up_penalty: Penalty when predicting DOWN but actual is UP (default: 1.5)
        down_penalty: Penalty when predicting UP but actual is DOWN (default: 1.0)
    """
    def __init__(self, up_penalty=1.5, down_penalty=1.0):
        super().__init__()
        self.mse = nn.MSELoss(reduction='none')
        self.up_penalty = up_penalty
        self.down_penalty = down_penalty
    
    def forward(self, pred, target, prev_values=None):
        """
        Args:
            pred: Predicted values. Shape: (batch, pred_len, features)
            target: Target values. Shape: (batch, pred_len, features)
            prev_values: Previous values for direction calculation. Shape: (batch,)
        
        Returns:
            Asymmetrically weighted loss value
        """
        # Base MSE loss per sample
        mse_loss = self.mse(pred, target).mean(dim=[1, 2])  # (batch,)
        
        if prev_values is None:
            return mse_loss.mean()
        
        # Get first prediction
        if len(pred.shape) == 3:
            pred_first = pred[:, 0, -1] if pred.shape[-1] > 1 else pred[:, 0, 0]
            target_first = target[:, 0, -1] if target.shape[-1] > 1 else target[:, 0, 0]
        else:
            pred_first = pred[:, 0]
            target_first = target[:, 0]
        
        if len(prev_values.shape) > 1:
            prev_values = prev_values[:, -1] if prev_values.shape[-1] > 1 else prev_values[:, 0]
        
        # Determine direction
        pred_up = pred_first > prev_values
        actual_up = target_first > prev_values
        
        # Create penalty weights
        weights = t.ones_like(mse_loss)
        # Predicted DOWN but actual UP -> missed profit
        weights = t.where(~pred_up & actual_up, weights * self.up_penalty, weights)
        # Predicted UP but actual DOWN -> incurred loss
        weights = t.where(pred_up & ~actual_up, weights * self.down_penalty, weights)
        
        return (mse_loss * weights).mean()


class WeightedDirectionalLoss(nn.Module):
    """
    Weighted Directional Loss: Weight direction errors by magnitude of move.
    
    Big moves matter more than small moves:
    - Getting a 5% move wrong is worse than getting a 0.1% move wrong
    - Focuses learning on capturing significant market movements
    
    Loss = MSE + direction_weight * (direction_error * (1 + magnitude_scale * |actual_return|))
    
    Args:
        direction_weight: Weight for directional component (default: 0.5)
        magnitude_scale: Scale factor for magnitude weighting (default: 10.0)
    """
    def __init__(self, direction_weight=0.5, magnitude_scale=10.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
        self.magnitude_scale = magnitude_scale
    
    def forward(self, pred, target, prev_values=None):
        """
        Args:
            pred: Predicted values. Shape: (batch, pred_len, features)
            target: Target values. Shape: (batch, pred_len, features)
            prev_values: Previous values for direction calculation. Shape: (batch,)
        
        Returns:
            Magnitude-weighted directional loss value
        """
        mse_loss = self.mse(pred, target)
        
        if prev_values is None or self.direction_weight == 0:
            return mse_loss
        
        if len(pred.shape) == 3:
            pred_first = pred[:, 0, -1] if pred.shape[-1] > 1 else pred[:, 0, 0]
            target_first = target[:, 0, -1] if target.shape[-1] > 1 else target[:, 0, 0]
        else:
            pred_first = pred[:, 0]
            target_first = target[:, 0]
        
        if len(prev_values.shape) > 1:
            prev_values = prev_values[:, -1] if prev_values.shape[-1] > 1 else prev_values[:, 0]
        
        # Compute returns (percentage change)
        pred_return = (pred_first - prev_values) / (t.abs(prev_values) + 1e-8)
        target_return = (target_first - prev_values) / (t.abs(prev_values) + 1e-8)
        
        # Direction match: +1 if same direction, -1 if different
        direction_match = t.sign(pred_return) * t.sign(target_return)
        
        # Weight by magnitude of actual move (bigger moves = bigger penalty if wrong)
        magnitude_weight = t.abs(target_return) * self.magnitude_scale
        
        # Penalize wrong direction, scaled by move magnitude
        # relu(-direction_match) = 0 if correct, 1 if wrong
        direction_loss = t.mean(t.relu(-direction_match) * (1 + magnitude_weight))
        
        return mse_loss + self.direction_weight * direction_loss


# Dictionary for easy access
STOCK_LOSSES = {
    'mse': nn.MSELoss,
    'directional': DirectionalLoss,
    'asymmetric': AsymmetricLoss,
    'weighted_directional': WeightedDirectionalLoss,
}
