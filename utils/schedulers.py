import math

class CosineWarmupScheduler:
    """Learning rate scheduler with warm-up and cosine annealing"""
    def __init__(self, optimizer, warmup_steps, max_steps):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self._step = 0 # Current step counter
        # Store base LRs for each param group
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]

    def step(self):
        """Update learning rate and return current value"""
        self._step += 1
        if self.warmup_steps > 0 and self._step < self.warmup_steps:
            # Linear warmup
            lr_scale = self._step / self.warmup_steps # Changed max(1,...) to ensure division by zero isn't an issue if warmup_steps=0
        else:
            # Cosine decay
            # Ensure max_steps is greater than warmup_steps to avoid division by zero/negative
            progress = (self._step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            # Clamp progress to [0, 1] to prevent issues if step exceeds max_steps
            progress = min(progress, 1.0)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))

        # Update learning rate for each parameter group
        for i, group in enumerate(self.optimizer.param_groups):
            group['lr'] = self.base_lrs[i] * lr_scale

        # Return current learning rate (optional, but can be useful)
        # return self.get_lr() # Remove if step shouldn't return LR

    def get_lr(self):
        """Get current learning rate"""
        # Ensure the optimizer's param_groups haven't changed structure
        # It's safer to return the calculated rate if structure might change,
        # but reading from optimizer is standard if structure is fixed.
        return [group['lr'] for group in self.optimizer.param_groups]

    # <<< --- ADD STATE DICT METHODS --- >>>
    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`."""
        return {
            'warmup_steps': self.warmup_steps,
            'max_steps': self.max_steps,
            '_step': self._step,
            'base_lrs': self.base_lrs,
            # Note: Optimizer state is saved separately
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.warmup_steps = state_dict['warmup_steps']
        self.max_steps = state_dict['max_steps']
        self._step = state_dict['_step']
        self.base_lrs = state_dict['base_lrs']
        # After loading state, apply the current LR based on loaded step
        # Re-run the logic from step() without incrementing _step
        current_step_for_lr = self._step
        if self.warmup_steps > 0 and current_step_for_lr < self.warmup_steps:
            lr_scale = current_step_for_lr / self.warmup_steps
        else:
            progress = (current_step_for_lr - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            lr_scale = 0.5 * (1.0 + math.cos(math.pi * progress))
        for i, group in enumerate(self.optimizer.param_groups):
             # Make sure base_lrs matches current optimizer structure
             if i < len(self.base_lrs):
                 group['lr'] = self.base_lrs[i] * lr_scale
             else:
                 print(f"Warning: Mismatch between loaded base_lrs ({len(self.base_lrs)}) and optimizer groups ({len(self.optimizer.param_groups)}). LR for group {i} not set by scheduler.")

