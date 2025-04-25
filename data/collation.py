import torch


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    # Filter out None/error samples
    # Keep your filtering logic as is
    valid_batch = [item for item in batch if item is not None and not isinstance(item, dict) or 'error' not in item]
    if not valid_batch:
        # Return dummy CPU tensors
        return {
            'pixel_values': torch.zeros(1, 3, 224, 224, device='cpu'),
            'labels': torch.zeros(1, 2, dtype=torch.long, device='cpu'),
            'base_character_indices': torch.zeros(1, 1, dtype=torch.long, device='cpu'),
            'diacritic_indices': torch.zeros(1, 1, dtype=torch.long, device='cpu'),
            'full_characters': [[""]],
            'word': [""]
        }
    batch = valid_batch

    # Extract data (ensure inputs are CPU tensors coming from dataset)
    pixel_values = []
    labels = []
    base_char_indices = []
    diacritic_indices = []
    full_characters = []
    words = []
    for sample in batch:
        # Ideally, ensure sample tensors are on CPU *before* collation
        pixel_values.append(sample['pixel_values']) # Assume CPU
        labels.append(sample['labels'])             # Assume CPU
        base_char_indices.append(sample['base_character_indices']) # Assume CPU
        diacritic_indices.append(sample['diacritic_indices'])       # Assume CPU
        full_characters.append(sample['full_characters'])
        words.append(sample.get('word', ''))

    # Stack pixel values
    pixel_values = torch.stack(pixel_values)

    # Pad labels (ensure padding happens on CPU)
    max_label_len = max(lab.size(0) for lab in labels)
    padded_labels = []
    for lab in labels:
        # <<< --- FIX: Use device='cpu' --- >>>
        padded = torch.full((max_label_len,), -100, dtype=torch.long, device='cpu')
        padded[:lab.size(0)] = lab.to('cpu') # Ensure input is CPU before copy
        padded_labels.append(padded)
    labels = torch.stack(padded_labels)

    # Pad base character indices (ensure padding happens on CPU)
    max_char_len = max(char.size(0) for char in base_char_indices)
    padded_base_chars = []
    for char in base_char_indices:
        # <<< --- FIX: Use device='cpu' --- >>>
        padded = torch.zeros((max_char_len,), dtype=torch.long, device='cpu')
        padded[:char.size(0)] = char.to('cpu') # Ensure input is CPU before copy
        padded_base_chars.append(padded)
    base_char_indices = torch.stack(padded_base_chars)

    # Pad diacritic indices (ensure padding happens on CPU)
    padded_diacritics = []
    for diac in diacritic_indices:
        # <<< --- FIX: Use device='cpu' --- >>>
        padded = torch.zeros((max_char_len,), dtype=torch.long, device='cpu')
        padded[:diac.size(0)] = diac.to('cpu') # Ensure input is CPU before copy
        padded_diacritics.append(padded)
    diacritic_indices = torch.stack(padded_diacritics)

    # Return CPU tensors
    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'base_character_indices': base_char_indices,
        'diacritic_indices': diacritic_indices,
        'full_characters': full_characters,
        'word': words
    }