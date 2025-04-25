import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import TrOCRProcessor, VisionEncoderDecoderModel, MBartForConditionalGeneration, AutoTokenizer
import unicodedata
import os
import json
from .submodules import _get_vocab_size, AttentionModule, Rethinking, DiacriticFeatureExtractor, SyllableProcessor, BartPhoCharacterProcessor 

class BartPhoVietOCR(nn.Module):
    """
    Enhanced Vietnamese OCR model that integrates:
    1. TrOCR's vision encoder
    2. BartPho's Vietnamese-specific decoder
    3. Character-level processing with diacritic awareness
    4. Rethinking module for prediction refinement
    """
    def __init__(self,
                 vision_encoder_name='microsoft/trocr-base-handwritten',
                 bartpho_name='vinai/bartpho-syllable-base',
                 max_seq_len=100,
                 rank=16,
                 config=None): # config is unused but kept for potential future use
        super().__init__()

        # Load models and tokenizers first to get config values
        self.vision_processor = TrOCRProcessor.from_pretrained(vision_encoder_name)
        self.bartpho_tokenizer = AutoTokenizer.from_pretrained(bartpho_name)

        # Load vision encoder from TrOCR
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(vision_encoder_name)
        self.vision_encoder = self.trocr_model.encoder

        # Load BartPho decoder
        self.bartpho_model = MBartForConditionalGeneration.from_pretrained(bartpho_name)
        self.text_decoder = self.bartpho_model.model.decoder
        self.lm_head = self.bartpho_model.lm_head

        # Get configuration values AFTER loading models
        self.hidden_size = self.text_decoder.config.hidden_size
        self.vocab_size = _get_vocab_size(self.bartpho_tokenizer) # Use helper
        self.max_seq_len = max_seq_len
        self.rank = rank

        # Load Vietnamese character vocabularies (needs self.hidden_size, etc)
        self.base_char_vocab = self._init_base_char_vocab()
        self.diacritic_vocab = self._init_diacritic_vocab()

        # Create adaptive layer to bridge vision encoder to text decoder
        self.adaptive_layer = nn.Linear(
            self.vision_encoder.config.hidden_size,
            self.text_decoder.config.hidden_size
        )

        # Initialize Rethinking module for prediction refinement
        self.rethinking = self._init_rethinking_module() # Needs self.vocab_size

        # Initialize BartPho-aware character processor
        self.character_processor = BartPhoCharacterProcessor(
            tokenizer=self.bartpho_tokenizer,
            hidden_size=self.hidden_size,
            base_char_vocab=self.base_char_vocab,
            diacritic_vocab=self.diacritic_vocab
        )

        # Token-Character attention module (kept for potential future use, currently unused in forward)
        # self.token_char_attention = AttentionModule(
        #     hidden_size=self.hidden_size,
        #     num_heads=8,
        #     dropout_rate=0.1,
        #     num_layers=2
        # )

        # Diacritic feature extractor
        self.diacritic_feature_extractor = DiacriticFeatureExtractor(
            hidden_size=self.hidden_size,
            diacritic_vocab=self.diacritic_vocab
        )

        # Syllable processor for Vietnamese syllables
        self.syllable_processor = SyllableProcessor(
            hidden_size=self.hidden_size,
            tokenizer=self.bartpho_tokenizer
        )

        # Character-level prediction heads
        self.base_char_head = nn.Linear(self.hidden_size, len(self.base_char_vocab))
        self.diacritic_head = nn.Linear(self.hidden_size, len(self.diacritic_vocab))

        # Context aggregation
        self.context_gru = nn.GRU(
            input_size=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.context_projection = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Special token IDs
        self.start_token_id = self.bartpho_tokenizer.bos_token_id
        self.pad_token_id = self.bartpho_tokenizer.pad_token_id
        self.end_token_id = self.bartpho_tokenizer.eos_token_id
        # Handle cases where pad_token_id might be None
        if self.pad_token_id is None:
             print("Warning: Tokenizer does not have a pad token ID. Using EOS token ID as fallback.")
             self.pad_token_id = self.end_token_id # Common fallback
        # Ensure cross entropy ignore_index is valid
        self.loss_ignore_index = self.pad_token_id if self.pad_token_id is not None else -100


        # Initialize weights
        self._init_weights()
        # Initialize remaining layers requiring specific dimensions
        self._init_projections() # Renamed from _init_alignment_networks for clarity

    def _init_projections(self):
        """Initialize projection layers."""
        # Get vision encoder and text decoder dimensions
        vision_dim = self.vision_encoder.config.hidden_size
        text_dim = self.text_decoder.config.hidden_size

        # Create special projection layer for adapting between these dimensions if not already done
        if not hasattr(self, 'adaptive_layer'):
             self.adaptive_layer = nn.Linear(vision_dim, text_dim)

        # Create special projection for decoder hidden states if needed for alignment
        # This might be created dynamically within alignment functions if dimensions vary
        # self.decoder_projection = nn.Linear(text_dim, self.hidden_size)
        pass # Alignment networks are now initialized within their respective modules

    def _init_base_char_vocab(self):
        """Initialize vocabulary of Vietnamese base characters"""
        # Added common punctuation and symbols often seen in OCR
        return [
            "<pad>", "a", "b", "c", "d", "e", "g", "h", "i", "k", "l", "m",
            "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y",
            "A", "B", "C", "D", "E", "G", "H", "I", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y",
            "đ", "Đ", "f", "F", "j", "J", "w", "W", "z", "Z",
            "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
            " ", ",", ".", "?", "!", ":", ";", "-", "_", "(", ")", "[", "]", "{", "}", "'", "\"", "/", "\\", "@", "#", "$", "%", "^", "&", "*", "+", "=", "<", ">", "|"
        ]

    def _init_diacritic_vocab(self):
        """Initialize vocabulary of Vietnamese diacritics"""
        return [
            "no_diacritic", "acute", "grave", "hook", "tilde", "dot",
            "circumflex", "breve", "horn", "stroke",
            "circumflex_grave", "circumflex_acute", "circumflex_tilde", "circumflex_hook", "circumflex_dot",
            "breve_grave", "breve_acute", "breve_tilde", "breve_hook", "breve_dot",
            "horn_grave", "horn_acute", "horn_tilde", "horn_hook", "horn_dot"
        ]

    def _init_rethinking_module(self):
        """Initialize the Rethinking module for prediction refinement"""
        # Ensure vocab_size is valid before initializing
        if not isinstance(self.vocab_size, int) or self.vocab_size <= 0:
            raise ValueError(f"Invalid vocab_size: {self.vocab_size}. Cannot initialize Rethinking module.")
        return Rethinking(
            vocab_size=self.vocab_size,
            rank=self.rank,
            max_seq_len=self.max_seq_len
        )

    def _init_weights(self):
        """Initialize weights with Xavier uniform distribution for better training"""
        for module in [self.adaptive_layer, self.base_char_head, self.diacritic_head,
                       self.context_projection]:
            # Check if module exists and has weights
            if hasattr(module, 'weight') and module.weight is not None:
                 nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)

    def _prepare_decoder_input_ids(self, batch_size, device):
        """Prepare decoder input IDs for autoregressive generation"""
        return torch.full((batch_size, 1), self.start_token_id, dtype=torch.long, device=device)

    def _prepare_encoder_attention_mask(self, hidden_states):
        """Create attention mask for the encoder outputs"""
        batch_size, seq_len = hidden_states.shape[:2]
        return torch.ones((batch_size, seq_len), dtype=torch.long, device=hidden_states.device)

    def forward(self, pixel_values, labels=None, decoder_input_ids=None):
            """
            Forward pass through the integrated model

            Args:
                pixel_values: Pixel values from image [batch_size, channels, height, width]
                labels: Optional target token IDs for training [batch_size, target_seq_len]
                        Should contain indices within [0, vocab_size-1] or -100 for padding/ignore.
                decoder_input_ids: Optional decoder input IDs [batch_size, input_seq_len]

            Returns:
                Dict containing loss, logits, and predictions
            """
            batch_size = pixel_values.shape[0]
            device = pixel_values.device

            # <<< --- DEBUG CHECK 1: INPUT LABELS (At start of forward) --- >>>
            if labels is not None:
                try:
                    # print(f"\n--- DEBUG INPUT LABELS (Start of Model Forward) ---")
                    # print(f"Input Labels Shape: {labels.shape}, dtype: {labels.dtype}")
                    unique_input_labels = torch.unique(labels)
                    min_val, max_val = labels.min().item(), labels.max().item()
                    # print(f"Input Labels min={min_val}, max={max_val}, unique={unique_input_labels.cpu().tolist()}")
                    # Check range including -100 but excluding ignore_index for OOB check
                    # Assumes self.vocab_size is correctly initialized
                    invalid_indices = unique_input_labels[
                        (unique_input_labels < 0) & (unique_input_labels != -100) | # Negative but not -100
                        (unique_input_labels >= self.vocab_size)                   # Out of bounds high
                    ]
                    if invalid_indices.numel() > 0:
                        print(f"!!! ERROR: Invalid Indices found in INPUT labels: {invalid_indices.cpu().tolist()} !!! (Vocab: {self.vocab_size})")
                    # else:
                    #     print(f"Input Label Indices appear valid w.r.t Vocab and -100.")
                    # print(f"--- End DEBUG INPUT LABELS ---\n")
                except Exception as e:
                    print(f"!!! ERROR during unique check for input labels: {e} !!!")
            # <<< --- END DEBUG CHECK 1 --- >>>

            # Process through vision encoder
            encoder_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)
            encoder_hidden_states = encoder_outputs.last_hidden_state # [B, SeqVision, DimVision]

            # Apply adaptive layer to match dimensions for BartPho decoder
            adapted_encoder_hidden_states = self.adaptive_layer(encoder_hidden_states) # [B, SeqVision, DimText]

            # Create encoder attention mask
            encoder_attention_mask = self._prepare_encoder_attention_mask(adapted_encoder_hidden_states)

            # Handle decoder input for training vs inference
            if labels is not None:
                # Training mode - use teacher forcing with labels
                decoder_input_ids = self._shift_right(labels) # [B, SeqText] - Shifts labels right, prepends BOS
                decoder_attention_mask = self._make_attention_mask(decoder_input_ids) # [B, SeqText] - Masks padding
            elif decoder_input_ids is None:
                # Inference mode without provided decoder inputs (e.g., starting generation)
                decoder_input_ids = self._prepare_decoder_input_ids(batch_size, device) # [B, 1] - BOS token
                decoder_attention_mask = None # Mask handled internally during generation or needs causal mask
            else:
                # Using provided decoder_input_ids (e.g., subsequent steps in generation)
                decoder_attention_mask = self._make_attention_mask(decoder_input_ids)


            # Forward through BartPho decoder
            decoder_outputs = self.text_decoder(
                input_ids=decoder_input_ids,
                attention_mask=decoder_attention_mask,
                encoder_hidden_states=adapted_encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                output_hidden_states=True, # Need hidden states for character processing
                return_dict=True
            )

            # Get decoder hidden states (last layer)
            decoder_hidden_states = decoder_outputs.last_hidden_state # [B, SeqText, HiddenSize]

            # Generate logits from decoder outputs
            logits = self.lm_head(decoder_hidden_states) # [B, SeqText, VocabSize]

            # Apply Rethinking module to refine logits
            refined_logits = logits + self.rethinking(logits) # [B, SeqText, VocabSize]

            # --- Calculate Token-Level Loss if labels are provided ---
            loss = None
            if labels is not None:
                # Ensure loss_ignore_index is valid (should be pad_token_id or -100)
                # loss_fct = nn.CrossEntropyLoss(ignore_index=self.loss_ignore_index)
                loss_fct = nn.CrossEntropyLoss(ignore_index=-100) # <<< Explicitly using -100 for loss ignore based on common practice
                # loss_fct = nn.CrossEntropyLoss(ignore_index=self.pad_token_id) # If you intend padding ID to be ignored

                # Shift logits and labels for standard seq2seq loss calculation
                # Predict token at pos t+1 using output from pos t
                shift_logits = refined_logits[..., :-1, :].contiguous() # [B, SeqText-1, VocabSize]
                shift_labels = labels[..., 1:].contiguous() # [B, SeqText-1] - Align target labels

                # <<< --- DEBUG CHECK 2: SHIFTED LABELS (Before Token Loss) --- >>>
                try:
                    # print(f"\n--- DEBUG SHIFTED LABELS (Before Token Loss) ---")
                    # print(f"Shift Logits Shape: {shift_logits.shape}") # Should be [B, SeqText-1, VocabSize]
                    # print(f"Shift Labels Shape: {shift_labels.shape}, dtype: {shift_labels.dtype}") # Should be [B, SeqText-1]
                    unique_targets = torch.unique(shift_labels)
                    min_val, max_val = shift_labels.min().item(), shift_labels.max().item()
                    #print(f"Shift Labels min={min_val}, max={max_val}, unique={unique_targets.cpu().tolist()}")

                    # Use the ACTUAL ignore_index used in loss_fct for checking validity
                    current_ignore_index = loss_fct.ignore_index
                    # print(f"Checking Shift Labels against VocabSize={self.vocab_size} and IgnoreIndex={current_ignore_index}")

                    # Check range excluding ignore_index for OOB check
                    invalid_indices_shifted = unique_targets[
                        (unique_targets < 0) & (unique_targets != current_ignore_index) | # Negative but not ignore_index
                        (unique_targets >= self.vocab_size)                                # Out of bounds high
                    ]
                    if invalid_indices_shifted.numel() > 0:
                        print(f"!!! ERROR: Invalid Indices found in SHIFTED labels: {invalid_indices_shifted.cpu().tolist()} !!!")
                    # else:
                    #     print(f"Shifted Label Indices appear valid (after ignore_index check).")
                    # print(f"--- End DEBUG SHIFTED LABELS ---\n")
                except Exception as e:
                    print(f"!!! ERROR during unique check for shift_labels: {e} !!!")
                # <<< --- END DEBUG CHECK 2 --- >>>

                # Calculate loss - Flatten the sequence dimension
                # Input to loss: Logits [N, C], Targets [N] where N = B * (SeqText-1)
                try:
                    loss = loss_fct(shift_logits.view(-1, self.vocab_size),
                                shift_labels.view(-1)) # <<< Error likely occurs here if indices are bad
                except Exception as loss_calc_e:
                    print(f"!!! ERROR during loss calculation: {loss_calc_e}")
                    # Optionally re-raise or handle, setting loss to None/0
                    loss = torch.tensor(0.0, device=device, requires_grad=True) # Placeholder to prevent downstream errors maybe?
                    # raise loss_calc_e # Or re-raise to stop execution


            # Determine output IDs for character processing (use labels if training, else predict)
            output_ids_for_char_processing = None
            if labels is not None:
                # Use labels during training for character processing alignment
                output_ids_for_char_processing = labels # Use original labels for char processing
            elif decoder_input_ids.size(1) > 1:
                # If decoder_input_ids were provided (longer than start token), use them
                output_ids_for_char_processing = decoder_input_ids
            else:
                # During inference starting from scratch, generate predictions first
                # Use the refined logits to get initial token predictions
                if refined_logits is not None: # Check if logits were computed
                    output_ids_for_char_processing = torch.argmax(refined_logits, dim=-1)


            # Process text at character level
            char_data = None
            char_level_results = {'base_char_logits': None, 'diacritic_logits': None}
            predictions = [] # Final text predictions list

            # Only attempt character processing if we have valid IDs
            if output_ids_for_char_processing is not None:
                try:
                    char_data = self.character_processor.process_text_char_level(
                        output_ids_for_char_processing,
                        decoder_hidden_states # Use last hidden state for potential alignment inside processor
                    )

                    # Process character-level data to get base/diacritic logits
                    char_level_results = self._process_character_level(char_data, decoder_hidden_states)

                    # Generate final predictions using character-level refinements
                    initial_token_preds = torch.argmax(refined_logits, dim=-1) if refined_logits is not None else output_ids_for_char_processing
                    predictions = self._generate_final_predictions(
                        char_data,
                        initial_token_preds, # Base token predictions
                        char_level_results['base_char_logits'],
                        char_level_results['diacritic_logits']
                    )
                except Exception as e:
                    # Fallback if character processing fails
                    print(f"Character-level processing error: {e}. Falling back to token-level prediction.")
                    # Generate predictions directly from refined logits if available
                    if refined_logits is not None:
                        pred_ids = torch.argmax(refined_logits, dim=-1)
                        predictions = []
                        for ids in pred_ids:
                            # Filter out padding and special tokens
                            valid_ids = ids[(ids != self.pad_token_id) & (ids != -100) & (ids != self.start_token_id) & (ids != self.end_token_id)]
                            text = self.bartpho_tokenizer.decode(valid_ids, skip_special_tokens=True)
                            predictions.append(text)
                    else:
                        predictions = ["<ERROR: No logits for fallback prediction>"] * batch_size

                    # Ensure char_level_results has None values if processing failed
                    char_level_results = {'base_char_logits': None, 'diacritic_logits': None}
            else:
                # Handle case where we couldn't get IDs for character processing
                print("Warning: Could not determine token IDs for character processing. Skipping character-level refinement.")
                # Generate fallback predictions directly from refined logits if available
                if refined_logits is not None:
                    pred_ids = torch.argmax(refined_logits, dim=-1)
                    predictions = []
                    for ids in pred_ids:
                        valid_ids = ids[(ids != self.pad_token_id) & (ids != -100) & (ids != self.start_token_id) & (ids != self.end_token_id)]
                        text = self.bartpho_tokenizer.decode(valid_ids, skip_special_tokens=True)
                        predictions.append(text)
                else:
                    predictions = ["<ERROR: No logits for prediction>"] * batch_size


            # Return comprehensive results
            return_dict = {
                'loss': loss, # This is the primary token-level loss
                'logits': refined_logits, # Final (refined) token logits
                'predictions': predictions, # Final text predictions (potentially character-corrected)
                'decoder_hidden_states': decoder_hidden_states, # For analysis/other heads
                'base_char_logits': char_level_results['base_char_logits'], # For base char loss/prediction
                'diacritic_logits': char_level_results['diacritic_logits'], # For diacritic loss/prediction
            }
            # Optionally include intermediate char_data if needed downstream
            # if char_data:
            #     return_dict['char_data'] = char_data

            return return_dict

    def _shift_right(self, input_ids):
        """Shift input IDs right for decoder input (prepend start token)"""
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
        shifted_input_ids[:, 0] = self.start_token_id

        # Ensure shifted IDs don't contain negative values intended for loss ignore (-100)
        # Replace any -100 with pad_token_id in the shifted input for the decoder itself.
        # The original labels tensor (with -100) is used for loss calculation later.
        if self.pad_token_id is not None:
            # Only replace -100 if pad_token_id is valid
             shifted_input_ids[shifted_input_ids == -100] = self.pad_token_id
        # else:
             # If no pad token, maybe raise error or leave -100? Leaving might cause issues.
             # Let's assume pad_token_id is always set, handled in __init__ fallback.

        return shifted_input_ids


    def _make_attention_mask(self, input_ids):
        """Create attention mask for padding tokens"""
        if self.pad_token_id is None:
             # If no pad token, assume all tokens should be attended to
             return torch.ones_like(input_ids, dtype=torch.long)
        # Mask is 1 for tokens that are NOT padding, 0 for padding tokens
        return (input_ids != self.pad_token_id).long()
    
    def _process_character_level(self, char_data, decoder_hidden_states):
        """Process character-level data for predictions"""
        # Extract character embeddings obtained from character_processor
        # These embeddings might already be aligned if decoder_hidden_states were passed to it.
        char_embeddings = char_data['char_embeddings'] # Shape: [B, MaxCharLen, HiddenSize]

        # Ensure all tensors are on the same device
        device = char_embeddings.device
        if decoder_hidden_states.device != device:
            decoder_hidden_states = decoder_hidden_states.to(device)

        # --- Syllable Processing ---
        # Process character embeddings with syllable awareness using token alignment info
        # Requires decoder_hidden_states to get token representations
        syllable_features = self.syllable_processor(
            char_embeddings,
            char_data['token_alignment'], # [B, MaxCharLen]
            decoder_hidden_states       # [B, SeqText, HiddenSize]
        ) # Output Shape: [B, MaxCharLen, HiddenSize]

        # --- Diacritic Feature Extraction ---
        # Enhance syllable-aware features with specific diacritic information
        diacritic_enhanced = self.diacritic_feature_extractor(
            syllable_features,          # Input features [B, MaxCharLen, HiddenSize]
            char_data['diacritic_indices'] # Diacritic info [B, MaxCharLen]
        ) # Output Shape: [B, MaxCharLen, HiddenSize]

        # --- Context Aggregation ---
        # Apply context aggregation using bidirectional GRU over the diacritic-enhanced features
        # GRU expects input: (batch, seq, feature)
        context_output, _ = self.context_gru(diacritic_enhanced) # Output Shape: [B, MaxCharLen, 2*HiddenSize]
        context_features = self.context_projection(context_output) # Output Shape: [B, MaxCharLen, HiddenSize]

        # Combine features: Add context features to the diacritic-enhanced features
        # Using addition as a simple fusion strategy
        enhanced_features = diacritic_enhanced + context_features # Shape: [B, MaxCharLen, HiddenSize]
        # print(f"--- Debug: Features before heads shape: {enhanced_features.shape} ---") # Should be [B, MaxCharLen, hidden_size]

        # --- Prediction Heads ---
        # Generate character-level predictions from the final enhanced features
        base_char_logits = self.base_char_head(enhanced_features) # [B, MaxCharLen, NumBaseChars]
        diacritic_logits = self.diacritic_head(enhanced_features) # [B, MaxCharLen, NumDiacritics]

        return {
            'base_char_logits': base_char_logits,
            'diacritic_logits': diacritic_logits
        }


    def _generate_final_predictions(self, char_data, token_ids, base_char_logits, diacritic_logits):
        """Generate final predictions with character-level refinements"""
        if base_char_logits is None or diacritic_logits is None:
            # Fallback if character logits are not available
            print("Warning: Character logits not available for final prediction generation. Using token-level decode.")
            predictions = []
            for ids in token_ids:
                valid_ids = ids[(ids != self.pad_token_id) & (ids != -100) & (ids != self.start_token_id) & (ids != self.end_token_id)]
                text = self.bartpho_tokenizer.decode(valid_ids, skip_special_tokens=True)
                predictions.append(text)
            return predictions

        batch_size = token_ids.size(0)
        predictions = []
        original_texts = char_data['texts'] # Use texts generated by character_processor
        max_char_len = char_data['max_char_len']

        # Ensure logits match the max_char_len used in char_data
        if base_char_logits.size(1) != max_char_len or diacritic_logits.size(1) != max_char_len:
             print(f"Warning: Logits length mismatch ({base_char_logits.size(1)}) vs char_data length ({max_char_len}). Truncating/Padding logits.")
             # Adjust logits shape (simple truncation or padding with a neutral value like 0)
             # This assumes the mismatch is due to padding differences between token/char levels
             target_len = max_char_len
             current_len_base = base_char_logits.size(1)
             current_len_diac = diacritic_logits.size(1)

             if current_len_base > target_len:
                 base_char_logits = base_char_logits[:, :target_len, :]
             elif current_len_base < target_len:
                 pad_width = target_len - current_len_base
                 # Pad with zeros (or a more neutral logit value if needed)
                 base_char_logits = F.pad(base_char_logits, (0, 0, 0, pad_width), "constant", 0)

             if current_len_diac > target_len:
                 diacritic_logits = diacritic_logits[:, :target_len, :]
             elif current_len_diac < target_len:
                 pad_width = target_len - current_len_diac
                 diacritic_logits = F.pad(diacritic_logits, (0, 0, 0, pad_width), "constant", 0)


        # Get character-level predictions and confidences
        base_char_preds = base_char_logits.argmax(dim=-1) # [B, MaxCharLen]
        diac_preds = diacritic_logits.argmax(dim=-1)     # [B, MaxCharLen]

        try:
            base_probs = F.softmax(base_char_logits, dim=-1)
            diac_probs = F.softmax(diacritic_logits, dim=-1)

            base_conf = torch.gather(
                base_probs, 2, # Gather along the vocab dimension
                base_char_preds.unsqueeze(-1)
            ).squeeze(-1) # [B, MaxCharLen]

            diac_conf = torch.gather(
                diac_probs, 2, # Gather along the vocab dimension
                diac_preds.unsqueeze(-1)
            ).squeeze(-1) # [B, MaxCharLen]
        except Exception as e:
            print(f"Error calculating confidence scores: {e}. Proceeding without confidence checks.")
            base_conf = torch.ones_like(base_char_preds, dtype=torch.float) # Assume full confidence
            diac_conf = torch.ones_like(diac_preds, dtype=torch.float)     # Assume full confidence


        for i in range(batch_size):
            original_text = original_texts[i]
            if not original_text: # Handle empty strings
                predictions.append("")
                continue

            # Create list to hold corrected characters
            corrected_chars = list(original_text)
            num_chars_in_text = len(original_text)

            # Apply character-level corrections up to the actual length of the text
            for char_pos in range(min(num_chars_in_text, max_char_len)): # Iterate only over valid positions
                # Get predictions and confidences for this position
                base_conf_score = base_conf[i, char_pos].item()
                diac_conf_score = diac_conf[i, char_pos].item()

                # Check confidence threshold (apply correction only if confident enough)
                confidence_threshold = 0.5 # Adjust as needed
                if base_conf_score < confidence_threshold or diac_conf_score < confidence_threshold:
                    continue # Skip correction if confidence is low

                # Get predicted base character and diacritic indices
                base_pred_idx = base_char_preds[i, char_pos].item()
                diac_pred_idx = diac_preds[i, char_pos].item()

                # Ensure indices are valid
                if 0 <= base_pred_idx < len(self.base_char_vocab) and \
                   0 <= diac_pred_idx < len(self.diacritic_vocab):
                    try:
                        # Get predicted base char and diacritic name
                        pred_base_char = self.base_char_vocab[base_pred_idx]
                        pred_diacritic = self.diacritic_vocab[diac_pred_idx]

                        # Get original character and decompose it for comparison/context
                        original_char = original_text[char_pos]
                        original_base, original_diacritic = self.character_processor.decompose_vietnamese_char(original_char)

                        # --- Correction Logic ---
                        # Apply correction if:
                        # 1. The original character is one that *can* take diacritics (e.g., a vowel).
                        # 2. The predicted base character matches the original base character OR prediction is very confident.
                        # 3. The predicted diacritic is different from the original.

                        # Basic check if the character is likely Vietnamese vowel/modifiable consonant
                        can_modify = self._is_vietnamese_modifiable(original_char)

                        if can_modify:
                            # Attempt to compose the new character
                            corrected_char = self.character_processor.compose_vietnamese_char(
                                pred_base_char, # Use predicted base char
                                pred_diacritic
                            )
                            # Only update if the composed character is different and valid
                            if corrected_char != original_char and corrected_char:
                                 # Simple correction: always apply if confident
                                 corrected_chars[char_pos] = corrected_char
                                 # More complex logic could be added here, e.g., checking if base char prediction matches original base

                    except IndexError:
                        print(f"Warning: Index out of bounds for vocab lookup at pos {char_pos} (Indices: base={base_pred_idx}, diac={diac_pred_idx})")
                        continue # Skip correction for this char
                    except Exception as e:
                        print(f"Error applying correction at pos {char_pos}: {e}")
                        continue # Skip correction for this char

            # Join the corrected characters
            predictions.append(''.join(corrected_chars))

        return predictions

    def _is_vietnamese_modifiable(self, char):
        """Check if a character is a Vietnamese vowel or consonant that can take diacritics (like đ)."""
        # Decompose to handle pre-composed characters
        try:
            norm_char = unicodedata.normalize('NFD', char.lower())
            base_char = norm_char[0]
            # Check if the base character is a standard vowel or 'd' (for đ)
            return base_char in "aeiouyd"
        except Exception:
             # Fallback for unexpected characters
             return char.lower() in "aeiouydđ"


    # Note: generate() method is simplified here. For full beam search etc.,
    # use the underlying self.bartpho_model.generate() with encoder outputs.
    def generate(self, pixel_values=None, encoder_outputs=None, max_length=None, **kwargs):
        """
        Generate text from image using BartPho decoder. Wraps the underlying model's generate.

        Args:
            pixel_values: Optional pixel values [batch_size, channels, height, width]
            encoder_outputs: Optional pre-computed encoder outputs (VisionEncoderDecoderOutput)
            max_length: Maximum sequence length to generate
            **kwargs: Additional arguments for the underlying generate method (e.g., num_beams)

        Returns:
            Tensor of token IDs [batch_size, seq_len]
        """
        if max_length is None:
            max_length = self.max_seq_len

        # Process through encoder if needed
        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("Either pixel_values or encoder_outputs must be provided")
            if pixel_values.device != self.vision_encoder.device:
                 pixel_values = pixel_values.to(self.vision_encoder.device)
            encoder_outputs = self.vision_encoder(pixel_values=pixel_values, return_dict=True)

        # Get encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state # [B, SeqVision, DimVision]

        # Apply adaptive layer to match dimensions for BartPho decoder
        adapted_encoder_hidden_states = self.adaptive_layer(encoder_hidden_states) # [B, SeqVision, DimText]

        # Create encoder attention mask
        encoder_attention_mask = self._prepare_encoder_attention_mask(adapted_encoder_hidden_states)

        # Use the underlying MBartForConditionalGeneration's generate method
        # We need to pass the adapted encoder hidden states and mask
        # The generate method handles the autoregressive loop, BOS token, EOS stopping etc.
        generated_ids = self.bartpho_model.generate(
            encoder_outputs=None, # Pass hidden states directly below
            encoder_hidden_states=adapted_encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            decoder_start_token_id=self.start_token_id,
            eos_token_id=self.end_token_id,
            pad_token_id=self.pad_token_id,
            **kwargs # Pass other generation config like num_beams
        )

        # Note: The Rethinking module is NOT applied during this standard generation.
        # Applying Rethinking during generation requires a custom generation loop
        # or modifying the logits processor list within the generate call.
        # For simplicity, this generate uses the base BartPho generation.
        # The `forward` method applies Rethinking for loss calculation and potentially inference if called directly.

        return generated_ids

    @torch.no_grad() # Ensure inference runs without gradients
    def inference(self, image):
        """
        Perform inference on a single image. Applies full model including char correction.

        Args:
            image: PIL Image object

        Returns:
            Dict containing the final corrected prediction.
        """
        self.eval() # Set model to evaluation mode

        # Process image
        pixel_values = self.vision_processor(image, return_tensors="pt").pixel_values
        device = next(self.parameters()).device # Get model's device
        pixel_values = pixel_values.to(device)

        # Perform a forward pass to get all intermediate results and corrected predictions
        # We don't need labels for inference.
        outputs = self.forward(pixel_values=pixel_values)

        # Get the first (and only) prediction from the batch
        final_prediction = outputs["predictions"][0] if outputs["predictions"] else ""

        # Optionally, decode the raw logits prediction for comparison
        raw_pred_ids = torch.argmax(outputs["logits"], dim=-1)[0]
        valid_ids = raw_pred_ids[(raw_pred_ids != self.pad_token_id) & (raw_pred_ids != -100) & (raw_pred_ids != self.start_token_id) & (raw_pred_ids != self.end_token_id)]
        raw_bartpho_prediction = self.bartpho_tokenizer.decode(valid_ids, skip_special_tokens=True)


        return {
            "prediction": final_prediction,
            "raw_bartpho_prediction": raw_bartpho_prediction,
            # Confidence calculation here is tricky, maybe average char confidences?
            # Placeholder: average max probability from token logits
            "confidence": torch.softmax(outputs["logits"], dim=-1).max(-1)[0].mean().item() if outputs["logits"] is not None else 0.0
        }

    # --- Save/Load Methods ---
    # Consider refining save/load to handle potential custom components like Rethinking state
    def save_pretrained(self, save_dir):
        """Save model components and state dict"""
        os.makedirs(save_dir, exist_ok=True)

        # Save base models/processors/tokenizers using their methods
        self.vision_processor.save_pretrained(save_dir) # Saves vision_processor related files
        self.vision_encoder.save_pretrained(os.path.join(save_dir, "vision_encoder"))
        self.bartpho_tokenizer.save_pretrained(os.path.join(save_dir, "bartpho_tokenizer"))
        # Saving the whole bartpho model might be large, consider saving only decoder + lm_head if needed
        # Or save the full model if required for easy reloading of the base structure
        self.bartpho_model.save_pretrained(os.path.join(save_dir, "bartpho_model"))

        # Save the custom model's state dict (includes adaptive layer, heads, rethinking, etc.)
        torch.save(self.state_dict(), os.path.join(save_dir, "custom_model_state.pt"))

        # Save vocabularies
        with open(os.path.join(save_dir, "base_char_vocab.json"), "w", encoding='utf8') as f:
            json.dump(self.base_char_vocab, f, ensure_ascii=False)
        with open(os.path.join(save_dir, "diacritic_vocab.json"), "w", encoding='utf8') as f:
            json.dump(self.diacritic_vocab, f, ensure_ascii=False)

        # Save config args like max_seq_len, rank if needed
        config_to_save = {"max_seq_len": self.max_seq_len, "rank": self.rank}
        with open(os.path.join(save_dir, "model_config.json"), "w", encoding='utf8') as f:
             json.dump(config_to_save, f)


        print(f"Model components saved to {save_dir}")

    @classmethod
    def from_pretrained(cls, load_dir):
        """Load model components and state dict"""
        if not os.path.isdir(load_dir):
            raise OSError(f"Directory not found: {load_dir}")

        # Load config args first
        config_path = os.path.join(load_dir, "model_config.json")
        model_config = {}
        if os.path.exists(config_path):
             with open(config_path, "r", encoding='utf8') as f:
                 model_config = json.load(f)

        # Determine base model names (assuming standard save format)
        # These might need to be stored/retrieved if non-default names were used
        vision_encoder_name = 'microsoft/trocr-base-handwritten' # Placeholder, ideally load from config if saved
        bartpho_name = 'vinai/bartpho-syllable-base'         # Placeholder

        # Instantiate the class with potentially loaded config
        model = cls(
             vision_encoder_name=vision_encoder_name, # Pass base names
             bartpho_name=bartpho_name,
             max_seq_len=model_config.get("max_seq_len", 100),
             rank=model_config.get("rank", 16)
        )

        # Load vocabularies before initializing components that depend on them
        try:
             with open(os.path.join(load_dir, "base_char_vocab.json"), "r", encoding='utf8') as f:
                 model.base_char_vocab = json.load(f)
             with open(os.path.join(load_dir, "diacritic_vocab.json"), "r", encoding='utf8') as f:
                 model.diacritic_vocab = json.load(f)
             # Re-initialize heads based on loaded vocab size
             model.base_char_head = nn.Linear(model.hidden_size, len(model.base_char_vocab))
             model.diacritic_head = nn.Linear(model.hidden_size, len(model.diacritic_vocab))
        except FileNotFoundError:
             print("Warning: Vocab files not found. Using default initialized vocabs/heads.")
             # Ensure heads match default vocab size if files are missing
             model.base_char_head = nn.Linear(model.hidden_size, len(model.base_char_vocab))
             model.diacritic_head = nn.Linear(model.hidden_size, len(model.diacritic_vocab))


        # Load the custom state dict
        state_dict_path = os.path.join(load_dir, "custom_model_state.pt")
        if os.path.exists(state_dict_path):
            try:
                 # Load state dict, mapping to the current device
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 model.load_state_dict(torch.load(state_dict_path, map_location=device), strict=False)
                 print(f"Loaded custom model state from {state_dict_path}")
            except Exception as e:
                 print(f"Warning: Could not load custom state dict from {state_dict_path}. Error: {e}. Model weights might be default.")
                 # Continue with potentially default initialized weights for custom parts
        else:
             print(f"Warning: Custom model state file not found: {state_dict_path}. Using default initialized weights for custom parts.")


        # Ensure model is on the correct device
        model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Model loaded from {load_dir}")
        return model

