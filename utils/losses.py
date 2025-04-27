
# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class FocalLoss(nn.Module):
#     """
#     Focal Loss implementation for handling class imbalance and focusing on hard examples.
#     Correctly handles ignore_index.

#     Paper: "Focal Loss for Dense Object Detection" - https://arxiv.org/abs/1708.02002
#     """
#     def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100): # Add ignore_index here
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#         self.ignore_index = ignore_index # Store ignore_index
#         # Initialize CrossEntropyLoss with reduction='none' and ignore_index
#         self.ce_loss_internal = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

#     def forward(self, inputs, targets):
#         """
#         Forward pass.

#         Args:
#             inputs (torch.Tensor): Predicted logits of shape (N, C) or (N, C, ...).
#             targets (torch.Tensor): Ground truth labels of shape (N,) or (N, ...).

#         Returns:
#             torch.Tensor: The calculated focal loss.
#         """
#         # Calculate standard cross entropy loss for each element.
#         # Elements where targets == ignore_index will have a loss of 0.
#         ce_loss = self.ce_loss(inputs, targets) # Shape: (N,) or (N, ...) matching targets shape

#         # Create a mask for valid (non-ignored) elements
#         mask = targets != self.ignore_index # Shape: (N,) or (N, ...)

#         # Check if there are any valid targets. If not, return 0 loss.
#         if not mask.any():
#             return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype)

#         # Select only the cross-entropy losses for valid targets
#         ce_loss_valid = ce_loss[mask]

#         # Calculate probabilities pt for valid targets only
#         # pt = exp(-ce_loss) -> this might be unstable if ce_loss is large
#         # Better: get log_pt directly from log_softmax and gather
#         log_softmax_inputs = F.log_softmax(inputs, dim=1)
#         log_pt_valid = log_softmax_inputs.view(-1, log_softmax_inputs.size(-1))[mask].gather(1, targets[mask].long().unsqueeze(1)).squeeze(1)
#         pt_valid = log_pt_valid.exp()


#         # Apply focal weighting: (1-pt)^gamma reduces the loss for well-classified examples
#         # Note: We use ce_loss_valid here which is -log_pt_valid
#         focal_term = (1 - pt_valid).pow(self.gamma)
#         focal_loss_valid = self.alpha * focal_term * ce_loss_valid # Loss for valid elements

#         # Apply reduction ONLY over valid elements
#         if self.reduction == 'mean':
#              # Average the loss over the number of VALID elements
#             return focal_loss_valid.sum() / mask.sum().clamp(min=1) # Use clamp to avoid div by zero
#         elif self.reduction == 'sum':
#             return focal_loss_valid.sum()
#         elif self.reduction == 'none':
#              # Return a loss tensor where ignored elements have loss 0
#              # Create a zero tensor matching the original targets shape
#              focal_loss_full = torch.zeros_like(targets, dtype=inputs.dtype)
#              # Place the calculated valid losses into the correct positions
#              focal_loss_full[mask] = focal_loss_valid
#              return focal_loss_full
#         else:
#              raise ValueError(f"Unsupported reduction type: {self.reduction}")

# # utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss implementation for handling class imbalance and focusing on hard examples.
    Correctly handles ignore_index.

    Paper: "Focal Loss for Dense Object Detection" - https://arxiv.org/abs/1708.02002
    """
    # --- Make sure reduction passed in __init__ is applied to internal CE ---
    # --- Or default internal CE to 'none' ---
    def __init__(self, alpha=1, gamma=2, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.final_reduction = reduction # Store the desired final reduction
        self.ignore_index = ignore_index # Store ignore_index
        # This ensures ce_loss below has per-element values before masking
        self.ce_loss_internal = nn.CrossEntropyLoss(reduction='none', ignore_index=self.ignore_index)

    def forward(self, inputs, targets):
        """
        Forward pass.

        Args:
            inputs (torch.Tensor): Predicted logits of shape (N, C) or (N, C, ...).
            targets (torch.Tensor): Ground truth labels of shape (N,) or (N, ...).

        Returns:
            torch.Tensor: The calculated focal loss.
        """
        if inputs.numel() == 0 or targets.numel() == 0:
             # Handle empty inputs gracefully
             if self.final_reduction == 'mean': return torch.tensor(0.0, device=inputs.device, requires_grad=True) # Return scalar 0 if mean
             elif self.final_reduction == 'sum': return torch.tensor(0.0, device=inputs.device, requires_grad=True) # Return scalar 0 if sum
             else: return torch.empty(0, device=inputs.device) # Return empty tensor if 'none'


        # Calculate standard cross entropy loss FOR EACH ELEMENT using internal CE.
        ce_loss_per_element = self.ce_loss_internal(inputs, targets) # Shape: (N,) or (N, ...) matching targets shape

        # Create a mask for valid (non-ignored) elements
        mask = targets != self.ignore_index # Shape: (N,) or (N, ...)

        # Check if there are any valid targets. If not, return 0 loss based on final reduction.
        if not mask.any():
             if self.final_reduction == 'mean' or self.final_reduction == 'sum':
                 # Return scalar zero that requires grad if inputs did
                 return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype, requires_grad=inputs.requires_grad)
             else: # reduction == 'none'
                 # Return tensor of zeros matching original targets shape
                 return torch.zeros_like(targets, dtype=inputs.dtype)


        # Select only the cross-entropy losses for valid targets using the mask
        # This indexing now works because ce_loss_per_element has the same shape as mask
        ce_loss_valid = ce_loss_per_element[mask]

        # Calculate probabilities pt for valid targets only
        log_softmax_inputs = F.log_softmax(inputs, dim=1)

        # Need to handle multi-dimensional inputs/targets correctly for gather
        # Flatten inputs/targets ONLY for the gather operation if necessary
        # Get the mask in the same flattened shape
        flat_mask = mask.view(-1)
        flat_log_softmax = log_softmax_inputs.view(-1, log_softmax_inputs.size(-1)) # [N_total, C]
        flat_targets = targets.view(-1) # [N_total]

        # Select valid elements using the flat mask
        valid_flat_log_softmax = flat_log_softmax[flat_mask] # [N_valid, C]
        valid_flat_targets = flat_targets[flat_mask].long().unsqueeze(1) # [N_valid, 1]

        # Gather the log probabilities corresponding to the true class for VALID elements
        if valid_flat_targets.numel() > 0: # Ensure there are elements to gather
             log_pt_valid = valid_flat_log_softmax.gather(1, valid_flat_targets).squeeze(1) # [N_valid]
             pt_valid = log_pt_valid.exp()
        else:
            # If somehow no valid targets after flattening (should be caught by mask.any() earlier, but safety check)
             if self.final_reduction == 'mean' or self.final_reduction == 'sum':
                 return torch.tensor(0.0, device=inputs.device, dtype=inputs.dtype, requires_grad=inputs.requires_grad)
             else:
                 return torch.zeros_like(targets, dtype=inputs.dtype)

        # Calculate focal loss components for valid elements
        focal_term = (1 - pt_valid).pow(self.gamma)
        focal_loss_valid = self.alpha * focal_term * ce_loss_valid # Loss for valid elements [N_valid]

        # --- Apply the desired FINAL reduction over VALID elements ---
        if self.final_reduction == 'mean':
            # Average the loss over the number of VALID elements
            # Use mask.sum() which counts true values (number of valid elements)
            num_valid = mask.sum().clamp(min=1) # Avoid division by zero
            return focal_loss_valid.sum() / num_valid
        elif self.final_reduction == 'sum':
            return focal_loss_valid.sum()
        elif self.final_reduction == 'none':
            # Return a loss tensor where ignored elements have loss 0
            focal_loss_full = torch.zeros_like(targets, dtype=inputs.dtype) # Match original targets shape
            # Place the calculated valid losses into the correct positions using the original mask
            focal_loss_full[mask] = focal_loss_valid
            return focal_loss_full
        else:
            raise ValueError(f"Unsupported reduction type: {self.final_reduction}")