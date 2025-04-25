import torch.optim as optim

def create_optimizer(model, learning_rate, weight_decay=0.01, discriminative_lr=False):
    """
    Create optimizer with optional discriminative learning rates
    
    Args:
        model: The model to optimize
        learning_rate: Base learning rate
        weight_decay: Weight decay factor
        discriminative_lr: Whether to use different LRs for different parts of the model
        
    Returns:
        Configured optimizer
    """
    if discriminative_lr:
        # Group parameters by component
        encoder_params = []
        decoder_params = []
        head_params = []
        other_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if 'vision_encoder' in name:
                encoder_params.append(param)
            elif 'text_decoder' in name or 'bartpho_model' in name:
                decoder_params.append(param)
            elif 'head' in name or 'char_head' in name or 'diacritic_head' in name:
                head_params.append(param)
            else:
                other_params.append(param)
        
        # Configure with different learning rates
        optimizer = optim.AdamW([
            {'params': encoder_params, 'lr': learning_rate * 0.1},  # Lower LR for pretrained encoder
            {'params': decoder_params, 'lr': learning_rate * 0.5},  # Medium LR for pretrained decoder
            {'params': head_params, 'lr': learning_rate * 1.5},     # Higher LR for task-specific heads
            {'params': other_params, 'lr': learning_rate}           # Default LR for other params
        ], weight_decay=weight_decay)
    else:
        # Simple optimizer with single learning rate
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    
    return optimizer
