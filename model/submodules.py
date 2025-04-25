import torch
import torch.nn as nn
import torch.nn.functional as F
import unicodedata


# Helper function to safely get vocab size
def _get_vocab_size(tokenizer):
    try:
        return tokenizer.vocab_size
    except AttributeError:
        # Handle cases where vocab_size might not be a direct attribute
        # or tokenizer might be None during early init stages.
        # Try getting the vocab dictionary length.
        try:
            return len(tokenizer.get_vocab())
        except Exception:
            # Fallback or raise a more informative error
            print("Warning: Could not determine vocab size from tokenizer.")
            return 50000 # A default large enough value, might need adjustment


class AttentionModule(nn.Module):
    """
    Enhanced self-attention module with multi-head attention and residual connections
    (Currently unused in BartPhoVietOCR forward path but kept for potential use)
    """
    def __init__(self, hidden_size, num_heads=8, dropout_rate=0.1, num_layers=2):
        super().__init__()
        self.num_layers = num_layers

        # Ensure number of heads divides hidden_size evenly
        if hidden_size % num_heads != 0:
            # Find largest divisor <= requested num_heads
            valid_num_heads = num_heads
            for h in range(num_heads, 0, -1):
                if hidden_size % h == 0:
                    valid_num_heads = h
                    break
            if valid_num_heads != num_heads:
                 print(f"Warning: Adjusting attention heads from {num_heads} to {valid_num_heads} to match hidden size {hidden_size}")
                 num_heads = valid_num_heads
            if hidden_size % num_heads != 0: # Should not happen now
                 raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}")


        # Create multiple attention layers
        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout_rate,
                batch_first=True # Expect (batch, seq, feature)
            ) for _ in range(num_layers)
        ])

        # Layer normalization for each attention layer output
        self.layer_norms_attn = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

        # Feed-forward layers after attention
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Dropout(dropout_rate),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(dropout_rate) # Dropout after final linear in FFN
            ) for _ in range(num_layers)
        ])

        # Layer normalization for each feed-forward layer output
        self.layer_norms_ff = nn.ModuleList([
            nn.LayerNorm(hidden_size) for _ in range(num_layers)
        ])

    def forward(self, x, attention_mask=None):
        """
        Args:
            x: Input tensor (batch_size, seq_len, hidden_size)
            attention_mask: Optional mask for attention (batch_size, seq_len)
                            or (batch_size, num_heads, seq_len, seq_len)
        Returns:
            Output tensor (batch_size, seq_len, hidden_size)
        """
        for i in range(self.num_layers):
            # Self-attention with residual connection
            # Note: nn.MultiheadAttention expects key_padding_mask (batch, key_seq)
            # or attn_mask (target_seq, source_seq) or (batch*num_heads, target_seq, source_seq)
            attn_output, _ = self.attn_layers[i](x, x, x, key_padding_mask=attention_mask, need_weights=False)
            x = self.layer_norms_attn[i](x + attn_output)  # Add & Norm

            # Feed-forward with residual connection
            ff_output = self.ff_layers[i](x)
            x = self.layer_norms_ff[i](x + ff_output)  # Add & Norm

        return x


class Rethinking(nn.Module):
    """
    Rethinking module from the VNHTR model for refining logits.
    Adds a learned residual based on self-attention over the vocabulary dimension.
    """
    def __init__(self, vocab_size, rank=16, max_seq_len=100, masked=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.rank = rank # Rank for the attention mechanism's intermediate projection
        self.block_size = max_seq_len # Max sequence length (T)
        self.masked = masked # Apply causal masking if True

        # Linear layer to project vocab dim (C) to rank*3 for Q, K, V
        self.c_attn = nn.Linear(self.vocab_size, self.rank * 3, bias=True)
        # Linear layer to project rank back to vocab dim (C)
        self.c_proj = nn.Linear(self.rank, self.vocab_size, bias=True)

        self.attn_dropout = nn.Dropout(0.0) # Dropout after softmax
        self.resid_dropout = nn.Dropout(0.1) # Dropout on the output residual
        self.gelu = nn.GELU()

    def forward(self, x):
        """
        Args:
            x: Input logits tensor (B, T, C) = (batch_size, seq_len, vocab_size)
        Returns:
            Residual tensor to be added to the input logits (B, T, C)
        """
        B, T, C = x.shape  # batch_size, block_size, vocab_size
        if C != self.vocab_size:
            raise ValueError(f"Input vocab size {C} does not match module vocab size {self.vocab_size}")

        # Project input logits to Q, K, V
        # x shape: (B, T, C) -> c_attn -> (B, T, rank*3)
        qkv = self.c_attn(x)
        # Split into Q, K, V each of shape (B, T, rank)
        q, k, v = qkv.split(self.rank, dim=2)

        # Calculate attention scores: Q @ K^T / sqrt(dk)
        # q shape: (B, T, rank), k.transpose shape: (B, rank, T)
        # att shape: (B, T, T)
        k_dim_sqrt = torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float, device=x.device)) # Use float tensor on correct device
        att = (q @ k.transpose(-2, -1)) * (1.0 / k_dim_sqrt)

        # Apply causal mask if needed
        if self.masked:
            # Create lower triangular mask (size T, T)
            # mask shape: (1, T, T) to allow broadcasting
            mask = torch.tril(torch.ones((T, T), device=x.device)).view(1, T, T) # Move mask to device
            # Fill positions where mask is 0 with -inf
            att = att.masked_fill(mask == 0, -float('inf'))

        # Apply softmax to get attention weights
        att = F.softmax(att, dim=-1)
        # Apply attention dropout
        att = self.attn_dropout(att)

        # Calculate output: Attention Weights @ V
        # att shape: (B, T, T), v shape: (B, T, rank)
        # y shape: (B, T, rank)
        y = att @ v

        # Project output back to vocab size
        # y shape: (B, T, rank) -> c_proj -> (B, T, C)
        y = self.c_proj(y)
        # Apply residual dropout
        y = self.resid_dropout(y)

        # Apply GELU activation to the residual
        # Note: Original paper might apply activation differently, check if needed.
        # This applies GELU to the final residual before adding back.
        return self.gelu(y)


class DiacriticFeatureExtractor(nn.Module):
    """Module to extract specialized features for Vietnamese diacritics"""
    def __init__(self, hidden_size, diacritic_vocab):
        super().__init__()
        self.diacritic_vocab = diacritic_vocab
        self.hidden_size = hidden_size
        self.diacritic_vocab_size = len(diacritic_vocab)

        # Ensure hidden_size is divisible by 4 for planned splits
        if hidden_size % 4 != 0:
             # Adjust intermediate dims or raise error
             # Simple approach: use hidden_size // 2 for both parts
             print(f"Warning: hidden_size {hidden_size} not divisible by 4. Adjusting DiacriticFeatureExtractor dims.")
             embed_dim = hidden_size // 2
             pos_dim = hidden_size // 2
             combiner_input_dim = hidden_size + hidden_size # Adjust combiner too
        else:
             embed_dim = hidden_size // 4
             pos_dim = hidden_size // 2 # Original intent seemed to be different dims
             combiner_input_dim = hidden_size + embed_dim + pos_dim # If concat all three parts

        # Let's try a simpler structure: Diacritic Embedding + Positional Encoding -> Combine -> Project
        diacritic_embed_dim = hidden_size // 4
        position_embed_dim = hidden_size // 4 # Make them equal for easier combination
        combined_input_dim = hidden_size + diacritic_embed_dim + position_embed_dim

        # Embedding for diacritic types
        self.diacritic_embeddings = nn.Embedding(self.diacritic_vocab_size, diacritic_embed_dim)

        # Position-aware encoding (applied to input hidden_states)
        # Projects hidden_states to a dimension suitable for combining with diacritic info
        self.position_encoding = nn.Sequential(
            nn.Linear(hidden_size, position_embed_dim),
            nn.GELU(),
            # No final linear here, just project
        )

        # Feature combiner: Takes original states + diacritic embeds + positional embeds
        self.feature_combiner = nn.Sequential(
            nn.Linear(combined_input_dim, hidden_size), # Project combined features back to hidden_size
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1) # Dropout after combination and projection
        )

        # Removed diacritic_transform and complex concatenations for simplification


    def forward(self, hidden_states, diacritic_indices=None):
        """
        Extract and combine diacritic-specific features.

        Args:
            hidden_states: Hidden states from model [batch_size, seq_len, hidden_size]
            diacritic_indices: Indices of diacritic types [batch_size, seq_len]

        Returns:
            Enhanced hidden states with diacritic features [batch_size, seq_len, hidden_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        device = hidden_states.device

        # Generate position features from hidden_states
        position_features = self.position_encoding(hidden_states) # [B, S, pos_embed_dim]

        # Get diacritic embeddings
        if diacritic_indices is None:
            # Use zeros or a learned 'no_diacritic' embedding during inference if needed
            # Using zeros is simpler here. Ensure correct dim.
            diacritic_embeds = torch.zeros(
                batch_size, seq_len, self.diacritic_embeddings.embedding_dim,
                device=device
            )
        else:
            # Handle potential size mismatch between hidden_states and diacritic_indices seq_len
            current_diac_len = diacritic_indices.size(1)
            if current_diac_len != seq_len:
                 # Pad or truncate diacritic_indices to match seq_len
                 if current_diac_len > seq_len:
                     diacritic_indices = diacritic_indices[:, :seq_len]
                 else:
                     pad_width = seq_len - current_diac_len
                     # Pad with 0 (assuming 0 is 'no_diacritic' or a safe default)
                     diacritic_indices = F.pad(diacritic_indices, (0, pad_width), "constant", 0)

            # Clamp indices to be within the valid range of the embedding layer
            diacritic_indices = torch.clamp(diacritic_indices, 0, self.diacritic_vocab_size - 1)
            diacritic_embeds = self.diacritic_embeddings(diacritic_indices) # [B, S, diacritic_embed_dim]


        # Combine original hidden states, diacritic embeddings, and position features
        combined_features = torch.cat([hidden_states, diacritic_embeds, position_features], dim=-1)
        # Expected shape: [B, S, hidden_size + diacritic_embed_dim + position_embed_dim]

        # Process combined features through the combiner network
        enhanced_features = self.feature_combiner(combined_features) # [B, S, hidden_size]

        # Add residual connection: Add the enhancement to the original hidden states
        return hidden_states + enhanced_features


class SyllableProcessor(nn.Module):
    """Process Vietnamese syllables using character embeddings and token context"""
    def __init__(self, hidden_size, tokenizer):
        super().__init__()
        self.hidden_size = hidden_size
        self.tokenizer = tokenizer # Needed? Maybe not directly.

        # Character-to-token alignment network
        # Takes concatenated [char_embed, token_embed] -> aligned_char_embed
        # Input size: hidden_size (char) + hidden_size (token) = hidden_size * 2
        self.alignment_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size), # Project combined back to hidden_size
            # nn.LayerNorm(hidden_size), # Norm might help stabilization
            # nn.GELU(),
            # nn.Dropout(0.1)
            # Simpler alignment: just a linear projection for now
        )

        # Syllable feature extraction (applied after alignment)
        # Takes concatenated [original_char_embed, aligned_char_embed] -> syllable_feature
        # Input size: hidden_size + hidden_size = hidden_size * 2
        self.syllable_feature_extractor = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            # nn.Linear(hidden_size, hidden_size) # Maybe remove last linear?
        )

    def forward(self, char_embeddings, token_alignment, decoder_hidden_states):
        """
        Process character embeddings with syllable awareness using token context.

        Args:
            char_embeddings: Character embeddings [batch_size, max_char_len, hidden_size]
            token_alignment: Token indices for each character [batch_size, max_char_len]
            decoder_hidden_states: Hidden states from decoder [batch_size, seq_len, hidden_size]

        Returns:
            Syllable-aware character embeddings [batch_size, max_char_len, hidden_size]
        """
        batch_size, max_char_len, hidden_size_char = char_embeddings.shape
        _, decoder_seq_len, hidden_size_decoder = decoder_hidden_states.shape
        device = char_embeddings.device

        # --- Device & Dimension Checks ---
        if decoder_hidden_states.device != device:
            decoder_hidden_states = decoder_hidden_states.to(device)
        if token_alignment.device != device:
            token_alignment = token_alignment.to(device)

        # Ensure decoder hidden states match the expected hidden_size for combination
        if hidden_size_char != hidden_size_decoder:
            # Project decoder states if dimensions mismatch (unlikely if using adaptive layer correctly)
            print(f"Warning: SyllableProcessor received decoder states ({hidden_size_decoder}) != char embeds ({hidden_size_char}). Projecting.")
            if not hasattr(self, 'decoder_proj'):
                 self.decoder_proj = nn.Linear(hidden_size_decoder, hidden_size_char).to(device)
            projected_decoder_states = self.decoder_proj(decoder_hidden_states)
        else:
            projected_decoder_states = decoder_hidden_states # Use directly

        # --- Alignment Step ---
        # Initialize aligned character embeddings (start with original embeddings)
        aligned_char_embeddings = char_embeddings.clone() # Start with original

        # Process each example in the batch
        for i in range(batch_size):
            # Gather token representations corresponding to each character
            # token_alignment[i] shape: [max_char_len]
            # Clamp indices to be safe
            valid_token_indices = torch.clamp(token_alignment[i], 0, decoder_seq_len - 1) # Clamp to valid range

            # Use gather to efficiently select token representations for each character position
            # projected_decoder_states[i] shape: [decoder_seq_len, hidden_size]
            # valid_token_indices shape: [max_char_len] -> needs unsqueeze/expand for gather
            # gather expects index shape: (max_char_len, 1) -> expand -> (max_char_len, hidden_size)
            token_reprs_for_chars = torch.gather(
                 projected_decoder_states[i], # Source tensor [decoder_seq_len, hidden_size]
                 0, # Dimension to gather along (sequence dim)
                 valid_token_indices.unsqueeze(-1).expand(-1, hidden_size_char) # Index tensor [max_char_len, hidden_size]
            ) # Result shape: [max_char_len, hidden_size]

            # Handle characters not aligned to any token (where token_alignment[i, j] was < 0 initially)
            no_token_mask = (token_alignment[i] < 0) # Mask for positions with no token alignment
            # Where mask is true, use zeros or original char embedding for token_repr part
            token_reprs_for_chars[no_token_mask] = 0.0 # Replace with zeros if no token

            # Combine character embedding with its corresponding token representation
            # char_embeddings[i] shape: [max_char_len, hidden_size]
            # token_reprs_for_chars shape: [max_char_len, hidden_size]
            combined = torch.cat([char_embeddings[i], token_reprs_for_chars], dim=-1) # Shape: [max_char_len, 2*hidden_size]

            # Apply alignment network
            aligned_chars_batch_i = self.alignment_network(combined) # Shape: [max_char_len, hidden_size]

            # Update the aligned embeddings for this batch item
            aligned_char_embeddings[i] = aligned_chars_batch_i


        # --- Feature Extraction Step ---
        # Combine original character embeddings with the aligned embeddings
        combined_for_extraction = torch.cat([char_embeddings, aligned_char_embeddings], dim=-1) # Shape: [B, MaxCharLen, 2*hidden_size]

        # Extract syllable features
        syllable_features = self.syllable_feature_extractor(combined_for_extraction) # Shape: [B, MaxCharLen, hidden_size]

        # --- Final Output ---
        # Combine with original embeddings via addition (residual connection)
        return char_embeddings + syllable_features


class BartPhoCharacterProcessor(nn.Module):
    """
    Enhanced character processor adapted for BartPho's syllable-based tokenization.
    Handles text-to-char mapping, alignment, and Vietnamese char decomposition/composition.
    """
    def __init__(self, tokenizer, hidden_size, base_char_vocab, diacritic_vocab):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab
        self.base_char_map = {char: i for i, char in enumerate(base_char_vocab)}
        self.diacritic_map = {diac: i for i, diac in enumerate(diacritic_vocab)}

        # Simplified character ID mapping (using ord()) - adjust size if needed
        self.char_vocab_size = 1000 # Max ord() value to map directly, others fallback
        self.char_embeddings = nn.Embedding(self.char_vocab_size, hidden_size, padding_idx=0) # Use index 0 for padding/unknown

        # Feature extraction network applied to char embeddings
        # Simplified: Just a linear layer + activation
        self.feature_extractor = nn.Sequential(
            # nn.Linear(hidden_size, hidden_size), # Removed redundancy
            # nn.LayerNorm(hidden_size),
            # nn.GELU(),
            # nn.Dropout(0.1),
            # nn.Linear(hidden_size, hidden_size), # Kept one projection
             nn.Linear(hidden_size, hidden_size),
             nn.GELU() # Added activation
        )

        # Character-to-token alignment network (used in _align_with_decoder)
        # Takes concatenated [char_feature, token_hidden_state]
        # Input: hidden_size * 2
        self.alignment_network = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            # nn.LayerNorm(hidden_size), # Optional normalization
            # nn.GELU(), # Optional activation
            # nn.Dropout(0.1) # Optional dropout
            # Simplified: Just linear projection
        )

        # Build Vietnamese character mappings
        self.char_to_base_diacritic = {}
        self.base_diacritic_to_char = {}
        self._build_vietnamese_mappings()

    def _build_vietnamese_mappings(self):
        """Build mappings for Vietnamese character decomposition and composition"""
        # Initialize essential mappings
        self.char_to_base_diacritic = {}
        self.base_diacritic_to_char = {}
        vowels = "aeiouy"
        diacritics_map = { # Map name to unicode combining char
            "acute": "\u0301", "grave": "\u0300", "hook": "\u0309",
            "tilde": "\u0303", "dot": "\u0323", "circumflex": "\u0302",
            "breve": "\u0306", "horn": "\u031b"
        }

        for vowel_base in vowels:
            for case_func in [str.lower, str.upper]:
                 vowel = case_func(vowel_base)
                 # Base vowel
                 self.char_to_base_diacritic[vowel] = (vowel, "no_diacritic")
                 self.base_diacritic_to_char[(vowel, "no_diacritic")] = vowel

                 # Simple diacritics (acute, grave, hook, tilde, dot)
                 for name, code in diacritics_map.items():
                      if name in ["circumflex", "breve", "horn"]: continue # Handle these separately
                      try:
                           composed = unicodedata.normalize('NFC', f"{vowel}{code}")
                           self.char_to_base_diacritic[composed] = (vowel, name)
                           self.base_diacritic_to_char[(vowel, name)] = composed
                      except Exception as e: print(f"Error composing {vowel}+{name}: {e}")

                 # Circumflex (â, ê, ô) - only on aeo
                 if vowel_base in "aeo":
                      try:
                           circ_code = diacritics_map["circumflex"]
                           circ_vowel = unicodedata.normalize('NFC', f"{vowel}{circ_code}")
                           self.char_to_base_diacritic[circ_vowel] = (vowel, "circumflex")
                           self.base_diacritic_to_char[(vowel, "circumflex")] = circ_vowel
                           # Circumflex + simple diacritics (ấ, ầ, ẩ, ẫ, ậ, etc.)
                           for name, code in diacritics_map.items():
                                if name in ["circumflex", "breve", "horn"]: continue
                                combo_name = f"circumflex_{name}"
                                try:
                                     composed = unicodedata.normalize('NFC', f"{vowel}{circ_code}{code}")
                                     self.char_to_base_diacritic[composed] = (vowel, combo_name)
                                     self.base_diacritic_to_char[(vowel, combo_name)] = composed
                                except Exception as e: print(f"Error composing {vowel}+circumflex+{name}: {e}")
                      except Exception as e: print(f"Error composing {vowel}+circumflex: {e}")


                 # Breve (ă) - only on a
                 if vowel_base == "a":
                      try:
                           breve_code = diacritics_map["breve"]
                           breve_vowel = unicodedata.normalize('NFC', f"{vowel}{breve_code}")
                           self.char_to_base_diacritic[breve_vowel] = (vowel, "breve")
                           self.base_diacritic_to_char[(vowel, "breve")] = breve_vowel
                           # Breve + simple diacritics (ắ, ằ, ẳ, ẵ, ặ)
                           for name, code in diacritics_map.items():
                                if name in ["circumflex", "breve", "horn"]: continue
                                combo_name = f"breve_{name}"
                                try:
                                     composed = unicodedata.normalize('NFC', f"{vowel}{breve_code}{code}")
                                     self.char_to_base_diacritic[composed] = (vowel, combo_name)
                                     self.base_diacritic_to_char[(vowel, combo_name)] = composed
                                except Exception as e: print(f"Error composing {vowel}+breve+{name}: {e}")
                      except Exception as e: print(f"Error composing {vowel}+breve: {e}")

                 # Horn (ư, ơ) - only on uo
                 if vowel_base in "uo":
                      try:
                           horn_code = diacritics_map["horn"]
                           horn_vowel = unicodedata.normalize('NFC', f"{vowel}{horn_code}")
                           self.char_to_base_diacritic[horn_vowel] = (vowel, "horn")
                           self.base_diacritic_to_char[(vowel, "horn")] = horn_vowel
                           # Horn + simple diacritics (ứ, ừ, ử, ữ, ự, etc.)
                           for name, code in diacritics_map.items():
                                if name in ["circumflex", "breve", "horn"]: continue
                                combo_name = f"horn_{name}"
                                try:
                                     composed = unicodedata.normalize('NFC', f"{vowel}{horn_code}{code}")
                                     self.char_to_base_diacritic[composed] = (vowel, combo_name)
                                     self.base_diacritic_to_char[(vowel, combo_name)] = composed
                                except Exception as e: print(f"Error composing {vowel}+horn+{name}: {e}")
                      except Exception as e: print(f"Error composing {vowel}+horn: {e}")

        # Add đ/Đ separately
        self.char_to_base_diacritic['đ'] = ('d', 'stroke')
        self.base_diacritic_to_char[('d', 'stroke')] = 'đ'
        self.char_to_base_diacritic['Đ'] = ('D', 'stroke')
        self.base_diacritic_to_char[('D', 'stroke')] = 'Đ'

        # print(f"Built Vietnamese character mappings with {len(self.char_to_base_diacritic)} chars.")


    def get_char_id(self, char):
        """Maps a character to an integer ID for embedding lookup."""
        char_ord = ord(char)
        if 0 < char_ord < self.char_vocab_size: # Use ord() if within range (excluding 0)
             return char_ord
        else:
             return 0 # Fallback to padding/unknown index 0


    def get_enhanced_token_spans(self, text, token_ids):
        """
        Maps BartPho token indices to character spans in the decoded text.
        Handles potential discrepancies and syllable merging.
        """
        spans = {} # token_idx -> (start_char, end_char)
        char_to_token = {} # char_pos -> token_idx
        current_char_pos = 0

        # Decode token IDs to tokens respecting special characters/prefixes
        tokens_with_ids = []
        valid_ids_for_decode = []
        original_indices = []

        # Filter and store original indices
        for i, token_id in enumerate(token_ids):
             tid_val = token_id.item() if hasattr(token_id, 'item') else int(token_id)
             # Keep track of original indices corresponding to valid tokens
             if tid_val != self.tokenizer.pad_token_id and \
                tid_val != -100 and \
                tid_val != self.tokenizer.bos_token_id and \
                tid_val != self.tokenizer.eos_token_id:
                   valid_ids_for_decode.append(tid_val)
                   original_indices.append(i) # Store original index i

        # Decode valid tokens
        if not valid_ids_for_decode:
            return {}, {} # No valid tokens

        try:
            # Use batch_decode for potentially better handling of merges/prefixes
            decoded_fragments = self.tokenizer.batch_decode(
                [[tid] for tid in valid_ids_for_decode], # Decode each ID individually first
                skip_special_tokens=False, # Keep special tokens initially? No, skip them.
                clean_up_tokenization_spaces=False # Important for BartPho space handling
            )
            # decoded_text_reconstructed = self.tokenizer.decode(valid_ids_for_decode, clean_up_tokenization_spaces=False)
            # print(f"Target Text: '{text}'")
            # print(f"Recon Text:  '{decoded_text_reconstructed}'")
            # print(f"Fragments: {decoded_fragments}")

        except Exception as e:
            print(f"Error during token decoding: {e}. Cannot determine spans.")
            return {}, {}

        # Match decoded fragments to the target text
        current_char_pos = 0
        for fragment_idx, fragment in enumerate(decoded_fragments):
            original_token_idx = original_indices[fragment_idx] # Get original index

            # Bartpho specific: remove prefix space if present for matching
            # The actual space in the text will be handled by advancing current_char_pos
            match_fragment = fragment.lstrip(' ') # Match without leading space if any

            if not match_fragment: # Skip empty fragments (like standalone space tokens sometimes)
                 # If fragment was just a space, advance pos if text has space
                 if fragment == ' ' and current_char_pos < len(text) and text[current_char_pos] == ' ':
                       current_char_pos += 1
                 continue

            # Search for the fragment in the text starting from current_char_pos
            try:
                # Find the start position of the match
                found_pos = text.find(match_fragment, current_char_pos)

                if found_pos != -1:
                    start_char = found_pos
                    end_char = start_char + len(match_fragment)
                    spans[original_token_idx] = (start_char, end_char)
                    for char_pos in range(start_char, end_char):
                        char_to_token[char_pos] = original_token_idx
                    # Update current_char_pos to the end of the matched fragment
                    current_char_pos = end_char
                else:
                     # Fragment not found - alignment failed for this token
                     # print(f"Warning: Could not align token fragment '{fragment}' (match:'{match_fragment}') in text '{text}' starting from pos {current_char_pos}")
                     # Attempt fuzzy match? Or just skip? Skip for now.
                     # Advance pos minimally to avoid getting stuck
                     current_char_pos += 1


            except Exception as e:
                print(f"Error during span finding for fragment '{fragment}': {e}")
                current_char_pos += 1 # Advance position

        return spans, char_to_token

    def process_text_char_level(self, text_ids, decoder_hidden_states=None):
        """
        Process text token IDs into character-level data with BartPho-specific handling.

        Args:
            text_ids: Token ids [batch_size, seq_len]
            decoder_hidden_states: Optional decoder hidden states for alignment [B, SeqText, HiddenSize]

        Returns:
            Dict with character-level data aligned to decoder sequence lengths
        """
        batch_size = text_ids.shape[0]
        device = text_ids.device

        # Decode texts using BartPho tokenizer - ensure clean decoding
        decoded_texts = []
        # Pad token ID needs careful handling if None
        pad_token_id_val = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else -999 # Use a value not in vocab

        for ids in text_ids:
            # Filter out padding, BOS, EOS, and potential -100 labels
            valid_ids = ids[(ids != pad_token_id_val) & \
                            (ids != self.tokenizer.bos_token_id) & \
                            (ids != self.tokenizer.eos_token_id) & \
                            (ids != -100)]

            if len(valid_ids) == 0:
                decoded_texts.append("")
                continue

            # Decode the valid sequence
            try:
                text = self.tokenizer.decode(valid_ids,
                                             skip_special_tokens=True,
                                             clean_up_tokenization_spaces=True # Clean up for final text representation
                                             )
                decoded_texts.append(text)
            except Exception as e:
                 print(f"Error decoding token IDs: {valid_ids}. Error: {e}")
                 decoded_texts.append("") # Append empty string on error


        # Determine maximum character length across the batch
        max_char_len = max(len(text) for text in decoded_texts) if decoded_texts else 0
        if max_char_len == 0: max_char_len = 1 # Ensure at least one position

        # --- Initialize Output Tensors ---
        # Use padding_idx (0) for char_ids padding
        char_ids = torch.full((batch_size, max_char_len), 0, dtype=torch.long, device=device)
        # Use 0 index (assuming it's <pad> or no_diacritic) for padding others
        base_char_indices = torch.full((batch_size, max_char_len), self.base_char_map.get("<pad>", 0), dtype=torch.long, device=device)
        diacritic_indices = torch.full((batch_size, max_char_len), self.diacritic_map.get("no_diacritic", 0), dtype=torch.long, device=device)
        # Use -1 for token alignment padding (indicates no token)
        token_alignment = torch.full((batch_size, max_char_len), -1, dtype=torch.long, device=device)

        # --- Process Each Example ---
        for i, (text, ids) in enumerate(zip(decoded_texts, text_ids)):
            if not text: continue # Skip empty texts

            # Get token spans and character-to-token mapping using the raw text_ids
            # We need the original token indices for alignment with decoder_hidden_states
            _, char_to_token_map = self.get_enhanced_token_spans(text, ids)

            # Process each character in the (cleaned) decoded text
            for j, char in enumerate(text):
                if j >= max_char_len: break # Stop if exceeding max length

                # 1. Character ID for embedding
                char_ids[i, j] = self.get_char_id(char)

                # 2. Token Alignment: Map char position back to original token index
                token_alignment[i, j] = char_to_token_map.get(j, -1) # Use -1 if no token mapped

                # 3. Decompose Vietnamese Character
                base_char, diacritic = self.decompose_vietnamese_char(char)

                # 4. Get Indices from Vocabularies (use .get for safety)
                base_idx = self.base_char_map.get(base_char, self.base_char_map.get("<pad>", 0)) # Fallback to pad/unknown
                diacritic_idx = self.diacritic_map.get(diacritic, self.diacritic_map.get("no_diacritic", 0)) # Fallback to no_diacritic

                base_char_indices[i, j] = base_idx
                diacritic_indices[i, j] = diacritic_idx


        # --- Character Embeddings and Feature Extraction ---
        char_embeddings = self.char_embeddings(char_ids) # [B, MaxCharLen, HiddenSize]
        char_features = self.feature_extractor(char_embeddings) # [B, MaxCharLen, HiddenSize]


        # --- Align with Decoder Hidden States (if provided) ---
        if decoder_hidden_states is not None:
            aligned_features = self._align_with_decoder(
                char_features,
                token_alignment,
                decoder_hidden_states
            )
        else:
            # If no decoder states, use the extracted char features directly
            aligned_features = char_features


        return {
            'char_embeddings': aligned_features, # These are the final features for downstream tasks
            'char_ids': char_ids, # Raw char IDs used for embedding lookup
            'base_char_indices': base_char_indices, # Indices for base char prediction head
            'diacritic_indices': diacritic_indices, # Indices for diacritic prediction head
            'token_alignment': token_alignment, # Mapping from char pos to token index
            'texts': decoded_texts, # The decoded texts for this batch
            'max_char_len': max_char_len # Max char length in the batch
        }


    def _align_with_decoder(self, char_features, token_alignment, decoder_hidden_states):
        """
        Align character features with decoder hidden states based on token alignment.
        Uses the alignment_network.
        """
        batch_size, max_char_len, hidden_size_char = char_features.shape
        _, decoder_seq_len, hidden_size_decoder = decoder_hidden_states.shape
        device = char_features.device

        # --- Device & Dimension Checks ---
        if decoder_hidden_states.device != device:
            decoder_hidden_states = decoder_hidden_states.to(device)
        if token_alignment.device != device:
            token_alignment = token_alignment.to(device)

        # Ensure decoder hidden states match the expected hidden_size for combination
        if hidden_size_char != hidden_size_decoder:
            # Project decoder states if dimensions mismatch
            print(f"Warning: Alignment received decoder states ({hidden_size_decoder}) != char features ({hidden_size_char}). Projecting.")
            # Use a dynamically created projection layer if needed
            if not hasattr(self, 'align_decoder_proj'):
                 self.align_decoder_proj = nn.Linear(hidden_size_decoder, hidden_size_char).to(device)
            projected_decoder_states = self.align_decoder_proj(decoder_hidden_states)
        else:
            projected_decoder_states = decoder_hidden_states # Use directly

        # --- Alignment via Network ---
        # Initialize aligned features (start with original char features as fallback)
        aligned_output_features = char_features.clone()

        # Process each batch item
        for i in range(batch_size):
             # Gather token representations corresponding to each character
             valid_token_indices = torch.clamp(token_alignment[i], 0, decoder_seq_len - 1)
             token_reprs_for_chars = torch.gather(
                 projected_decoder_states[i], 0,
                 valid_token_indices.unsqueeze(-1).expand(-1, hidden_size_char)
             )
             # Handle characters not aligned to any token
             no_token_mask = (token_alignment[i] < 0)
             token_reprs_for_chars[no_token_mask] = 0.0 # Use zeros for unaligned

             # Combine character feature with its corresponding token representation
             combined = torch.cat([char_features[i], token_reprs_for_chars], dim=-1) # Shape: [max_char_len, 2*hidden_size]

             # Apply alignment network
             aligned_batch_i = self.alignment_network(combined) # Shape: [max_char_len, hidden_size]

             # Apply alignment only where a token was present (optional, could refine fallback)
             # aligned_output_features[i][~no_token_mask] = aligned_batch_i[~no_token_mask]
             # Simpler: update all positions with the aligned result
             aligned_output_features[i] = aligned_batch_i


        return aligned_output_features


    def decompose_vietnamese_char(self, char):
        """
        Decompose a Vietnamese character into base character and diacritic mark.
        Uses the pre-built mapping first, then falls back to Unicode normalization.
        """
        # 1. Try direct mapping
        if char in self.char_to_base_diacritic:
            return self.char_to_base_diacritic[char]

        # 2. Fallback to Unicode normalization (NFD - Canonical Decomposition)
        try:
            norm_char = unicodedata.normalize('NFD', char)
            base_char = norm_char[0]
            diacritic = 'no_diacritic' # Default

            if len(norm_char) > 1:
                 # Collect combining characters (diacritics)
                 combining_chars = [c for c in norm_char[1:] if unicodedata.combining(c)]
                 # Try to map combinations back to known names (simplified logic)
                 if '\u0302' in combining_chars: # Circumflex
                      if '\u0301' in combining_chars: diacritic = 'circumflex_acute'
                      elif '\u0300' in combining_chars: diacritic = 'circumflex_grave'
                      elif '\u0309' in combining_chars: diacritic = 'circumflex_hook'
                      elif '\u0303' in combining_chars: diacritic = 'circumflex_tilde'
                      elif '\u0323' in combining_chars: diacritic = 'circumflex_dot'
                      else: diacritic = 'circumflex'
                 elif '\u0306' in combining_chars: # Breve
                      if '\u0301' in combining_chars: diacritic = 'breve_acute'
                      # ... add other breve combinations ...
                      else: diacritic = 'breve'
                 elif '\u031b' in combining_chars: # Horn
                      if '\u0301' in combining_chars: diacritic = 'horn_acute'
                      # ... add other horn combinations ...
                      else: diacritic = 'horn'
                 elif '\u0301' in combining_chars: diacritic = 'acute'
                 elif '\u0300' in combining_chars: diacritic = 'grave'
                 elif '\u0309' in combining_chars: diacritic = 'hook'
                 elif '\u0303' in combining_chars: diacritic = 'tilde'
                 elif '\u0323' in combining_chars: diacritic = 'dot'
                 elif '\u0336' in combining_chars: # Long stroke overlay (might be relevant?)
                      diacritic = 'stroke' # Or map differently?
                 # Handle 'đ' case (d + stroke)
                 if base_char == 'd' and '\u0336' in combining_chars: # Check specific stroke for d
                      diacritic = 'stroke'

            # Special case for 'đ' if not caught above
            if char == 'đ': return ('d', 'stroke')
            if char == 'Đ': return ('D', 'stroke')

            # Return decomposed parts (might be approximate if mapping fails)
            return base_char, diacritic

        except Exception as e:
            # print(f"Error decomposing char '{char}': {e}. Returning base char.")
            # Fallback if normalization fails
            return char, 'no_diacritic'


    def compose_vietnamese_char(self, base_char, diacritic_name):
        """
        Compose a Vietnamese character from base character and diacritic name.
        Uses the pre-built mapping first, then falls back to Unicode composition.
        """
        # 1. Handle 'no_diacritic' or empty diacritic
        if not diacritic_name or diacritic_name.lower() == 'no_diacritic':
            return base_char

        # 2. Try direct mapping
        key = (base_char, diacritic_name)
        if key in self.base_diacritic_to_char:
            return self.base_diacritic_to_char[key]

        # 3. Fallback to Unicode composition (less reliable for complex cases)
        diacritic_map_rev = { # Map name back to Unicode combining characters
            'grave': '\u0300', 'acute': '\u0301', 'tilde': '\u0303', 'hook': '\u0309', 'dot': '\u0323',
            'circumflex': '\u0302', 'breve': '\u0306', 'horn': '\u031b', 'stroke': '\u0336', # Use standard stroke overlay
            'circumflex_grave': '\u0302\u0300', 'circumflex_acute': '\u0302\u0301', 'circumflex_tilde': '\u0302\u0303',
            'circumflex_hook': '\u0302\u0309', 'circumflex_dot': '\u0302\u0323',
            'breve_grave': '\u0306\u0300', 'breve_acute': '\u0306\u0301', 'breve_tilde': '\u0306\u0303',
            'breve_hook': '\u0306\u0309', 'breve_dot': '\u0306\u0323',
            'horn_grave': '\u031b\u0300', 'horn_acute': '\u031b\u0301', 'horn_tilde': '\u031b\u0303',
            'horn_hook': '\u031b\u0309', 'horn_dot': '\u031b\u0323',
        }
        diac_chars = diacritic_map_rev.get(diacritic_name.lower(), '')
        if not diac_chars:
             # print(f"Warning: Unknown diacritic name '{diacritic_name}' for composition.")
             return base_char # Return base if diacritic unknown

        try:
            combined = base_char + diac_chars
            # Normalize using NFC (Canonical Composition)
            normalized = unicodedata.normalize('NFC', combined)
            return normalized
        except Exception as e:
             # print(f"Error composing '{base_char}' + '{diacritic_name}': {e}")
             return base_char # Fallback to base character on error