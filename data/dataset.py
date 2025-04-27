# improved_bartpho_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import unicodedata
import ast
import logging

class ImprovedBartPhoDataset(Dataset):
    """Enhanced dataset class for BartPhoVietOCR with improved error handling and decomposition"""
    
    def __init__(self, hf_dataset, processor, base_char_vocab, diacritic_vocab):
        """
        Initialize the dataset
        
        Args:
            hf_dataset: HuggingFace dataset or similar
            processor: Vision processor from the BartPhoVietOCR model
            base_char_vocab: Vocabulary of base characters
            diacritic_vocab: Vocabulary of diacritics
        """
        self.dataset = hf_dataset
        self.processor = processor
        self.base_char_vocab = base_char_vocab
        self.diacritic_vocab = diacritic_vocab
        
        # Set up logging
        self.logger = logging.getLogger("ImprovedBartPhoDataset")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(handler)
        
        # Initialize Vietnamese character mappings
        self.char_to_base_diacritic = {}
        self.base_diacritic_to_char = {}
        self._build_vietnamese_mappings()
        
        # Cache for decomposed characters to avoid repeated calculations
        self.decomposition_cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            # Get example from dataset
            example = self.dataset[idx]
            
            # Extract image
            image = example['image']
            
            # Handle different image types
            pixel_values = self._process_image(image)
            
            # Extract character data with fallbacks
            word, char_data = self._extract_character_data(example)
            
            # Process with BartPho tokenizer
            labels = self._tokenize_text(word)
            
            # Prepare tensor outputs
            base_char_indices = torch.tensor(char_data['base_char_indices'])
            diacritic_indices = torch.tensor(char_data['diacritic_indices'])
            
            # Construct final sample
            return {
                'pixel_values': pixel_values,
                'labels': labels,
                'base_character_indices': base_char_indices,
                'diacritic_indices': diacritic_indices,
                'full_characters': char_data['full_chars'],
                'word': word
            }
        except Exception as e:
            self.logger.error(f"Error processing item {idx}: {e}")
            # Return a dummy sample that won't break training
            return self._create_dummy_sample()
    
    def _process_image(self, image):
        """Process image into tensor format"""
        try:
            # Handle PIL images
            if isinstance(image, Image.Image):
                # Convert PIL image to RGB to ensure 3 channels
                image = image.convert('RGB')
            elif isinstance(image, np.ndarray):
                # Handle numpy arrays
                if image.ndim == 2:
                    # Add channel dimension
                    image = np.expand_dims(image, axis=2)
                    # Repeat to make RGB
                    image = np.repeat(image, 3, axis=2)
                # Convert to PIL
                image = Image.fromarray(image.astype(np.uint8))
            
            # Process with vision processor
            encoding = self.processor(image, return_tensors='pt')
            return encoding.pixel_values.squeeze()
        except Exception as e:
            self.logger.warning(f"Error processing image: {e}. Using fallback.")
            # Create a fallback image
            dummy_image = Image.new('RGB', (32, 32), color='white')
            encoding = self.processor(dummy_image, return_tensors='pt')
            return encoding.pixel_values.squeeze()
    
    def _extract_character_data(self, example):
        """Extract character data with various fallbacks"""
        # Initialize with defaults
        full_chars = []
        base_chars = []
        diacritics = []
        word = ""
        
        # Try to extract full characters
        if 'full_character' in example:
            full_chars = self._parse_if_string_list(example['full_character'])
            if isinstance(full_chars, str):
                full_chars = [c for c in full_chars]
        # Fallback to label
        elif 'label' in example:
            word = example['label']
            full_chars = [c for c in word]
        
        # Set word from full characters if not already set
        if not word and full_chars:
            word = ''.join(full_chars) if isinstance(full_chars, list) else full_chars
        
        # Try to get base characters and diacritics
        if 'base_character' in example and 'diacritic_type' in example:
            base_chars = self._parse_if_string_list(example['base_character'])
            diacritics = self._parse_if_string_list(example['diacritic_type'])
            
            # Ensure they are lists
            if isinstance(base_chars, str):
                base_chars = [base_chars]
            if isinstance(diacritics, str):
                diacritics = [diacritics]
                
            # Ensure lengths match
            if len(base_chars) != len(diacritics):
                if len(base_chars) > len(diacritics):
                    diacritics.extend(['no_diacritic'] * (len(base_chars) - len(diacritics)))
                else:
                    base_chars.extend([''] * (len(diacritics) - len(base_chars)))
        
        # If we don't have base_chars/diacritics but have full_chars, decompose them
        elif full_chars:
            base_chars = []
            diacritics = []
            for char in full_chars:
                if isinstance(char, str) and len(char) > 0:
                    base, diac = self._decompose_vietnamese_char(char)
                    base_chars.append(base)
                    diacritics.append(diac)
                else:
                    base_chars.append('')
                    diacritics.append('no_diacritic')
        
        # Convert to indices
        base_char_indices = []
        for char in base_chars:
            try:
                index = self.base_char_vocab.index(char)
            except ValueError:
                self.logger.warning(f"Character '{char}' not found in base_char_vocab")
                index = 0  # Default to first character as fallback
            base_char_indices.append(index)
        
        diacritic_indices = []
        for diac in diacritics:
            if (diac == 'none') or (diac == None):
                diac = 'no_diacritic'
            try:
                index = self.diacritic_vocab.index(diac)
            except ValueError:
                self.logger.warning(f"Diacritic '{diac}' not found in diacritic_vocab")
                index = 0  # Default to first diacritic as fallback
            diacritic_indices.append(index)
        
        # Return the extracted data
        return word, {
            'full_chars': full_chars,
            'base_chars': base_chars,
            'diacritics': diacritics,
            'base_char_indices': base_char_indices,
            'diacritic_indices': diacritic_indices
        }
    
    def _tokenize_text(self, text):
        """Tokenize text using BartPho tokenizer"""
        try:
            # For simplicity, we'll import the tokenizer here since it's used infrequently
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("vinai/bartpho-syllable-base")
            return tokenizer(text, return_tensors='pt').input_ids.squeeze()
        except Exception as e:
            self.logger.error(f"Error tokenizing text '{text}': {e}")
            # Return a simple tensor with BOS token
            return torch.tensor([1], dtype=torch.long)  # 1 is typically BOS token
    
    def _create_dummy_sample(self):
        """Create a dummy sample for error cases"""
        # Create dummy tensors
        pixel_values = torch.zeros(3, 224, 224)
        labels = torch.tensor([1, 2, 3])  # Simple sequence with BOS, some token, EOS
        base_char_indices = torch.tensor([0])
        diacritic_indices = torch.tensor([0])
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'base_character_indices': base_char_indices,
            'diacritic_indices': diacritic_indices,
            'full_characters': ["?"],
            'word': "?"
        }
    
    def _parse_if_string_list(self, value):
        """Parse string representations of lists into actual lists"""
        if isinstance(value, str) and value.startswith('[') and value.endswith(']'):
            try:
                # Safely parse the string into a Python object
                return ast.literal_eval(value)
            except (SyntaxError, ValueError):
                # If parsing fails, return the original string
                return value
        return value
    
    def _build_vietnamese_mappings(self):
        """Build mappings for Vietnamese character decomposition and composition"""
        # Initialize essential mappings
        self.char_to_base_diacritic = {}
        self.base_diacritic_to_char = {}
        
        # Define basic vowels that can take diacritics
        vowels = "aeiouy"
        
        # Define diacritic combinations
        diacritics = [
            "no_diacritic", "acute", "grave", "hook", "tilde", "dot", 
            "circumflex", "breve", "horn"
        ]
        
        # For each vowel and diacritic, create mapping
        for vowel in vowels:
            # Base vowel without diacritics
            self.char_to_base_diacritic[vowel] = (vowel, "no_diacritic")
            self.char_to_base_diacritic[vowel.upper()] = (vowel.upper(), "no_diacritic")
            
            # Build mappings for common Vietnamese diacritical combinations
            try:
                # Acute accent (á, é, etc.)
                accented = unicodedata.normalize('NFC', f"{vowel}\u0301")
                self.char_to_base_diacritic[accented] = (vowel, "acute")
                self.base_diacritic_to_char[(vowel, "acute")] = accented
                self.char_to_base_diacritic[accented.upper()] = (vowel.upper(), "acute")
                self.base_diacritic_to_char[(vowel.upper(), "acute")] = accented.upper()
                
                # Grave accent (à, è, etc.)
                accented = unicodedata.normalize('NFC', f"{vowel}\u0300")
                self.char_to_base_diacritic[accented] = (vowel, "grave")
                self.base_diacritic_to_char[(vowel, "grave")] = accented
                self.char_to_base_diacritic[accented.upper()] = (vowel.upper(), "grave")
                self.base_diacritic_to_char[(vowel.upper(), "grave")] = accented.upper()
                
                # Hook (ả, ẻ, etc.)
                accented = unicodedata.normalize('NFC', f"{vowel}\u0309")
                self.char_to_base_diacritic[accented] = (vowel, "hook")
                self.base_diacritic_to_char[(vowel, "hook")] = accented
                self.char_to_base_diacritic[accented.upper()] = (vowel.upper(), "hook")
                self.base_diacritic_to_char[(vowel.upper(), "hook")] = accented.upper()
                
                # Tilde (ã, ẽ, etc.)
                accented = unicodedata.normalize('NFC', f"{vowel}\u0303")
                self.char_to_base_diacritic[accented] = (vowel, "tilde")
                self.base_diacritic_to_char[(vowel, "tilde")] = accented
                self.char_to_base_diacritic[accented.upper()] = (vowel.upper(), "tilde")
                self.base_diacritic_to_char[(vowel.upper(), "tilde")] = accented.upper()
                
                # Dot (ạ, ẹ, etc.)
                accented = unicodedata.normalize('NFC', f"{vowel}\u0323")
                self.char_to_base_diacritic[accented] = (vowel, "dot")
                self.base_diacritic_to_char[(vowel, "dot")] = accented
                self.char_to_base_diacritic[accented.upper()] = (vowel.upper(), "dot")
                self.base_diacritic_to_char[(vowel.upper(), "dot")] = accented.upper()
                
                # Add circumflex for a, e, o (â, ê, ô)
                if vowel in "aeo":
                    accented = unicodedata.normalize('NFC', f"{vowel}\u0302")
                    self.char_to_base_diacritic[accented] = (vowel, "circumflex")
                    self.base_diacritic_to_char[(vowel, "circumflex")] = accented
                    self.char_to_base_diacritic[accented.upper()] = (vowel.upper(), "circumflex")
                    self.base_diacritic_to_char[(vowel.upper(), "circumflex")] = accented.upper()
                    
                    # Combinations with circumflex
                    for diac, code in [("acute", "\u0301"), ("grave", "\u0300"), 
                                       ("hook", "\u0309"), ("tilde", "\u0303"), ("dot", "\u0323")]:
                        combo = unicodedata.normalize('NFC', f"{vowel}\u0302{code}")
                        combo_name = f"circumflex_{diac}"
                        self.char_to_base_diacritic[combo] = (vowel, combo_name)
                        self.base_diacritic_to_char[(vowel, combo_name)] = combo
                        self.char_to_base_diacritic[combo.upper()] = (vowel.upper(), combo_name)
                        self.base_diacritic_to_char[(vowel.upper(), combo_name)] = combo.upper()
            except Exception as e:
                self.logger.warning(f"Error building Vietnamese character mappings: {e}")
        
        # Add đ/Đ
        self.char_to_base_diacritic['đ'] = ('d', 'stroke')
        self.base_diacritic_to_char[('d', 'stroke')] = 'đ'
        self.char_to_base_diacritic['Đ'] = ('D', 'stroke')
        self.base_diacritic_to_char[('D', 'stroke')] = 'Đ'
        
        # Add consonants and numbers without diacritics
        consonants = "bcdfghjklmnpqrstvwxz"
        for char in consonants:
            if char not in self.char_to_base_diacritic:
                self.char_to_base_diacritic[char] = (char, "no_diacritic")
                self.char_to_base_diacritic[char.upper()] = (char.upper(), "no_diacritic")
        
        for num in "0123456789":
            self.char_to_base_diacritic[num] = (num, "no_diacritic")
    
    def _decompose_vietnamese_char(self, char):
        """
        Decompose a Vietnamese character into base character and diacritic mark
        with caching for better performance
        """
        # Check cache first
        if char in self.decomposition_cache:
            return self.decomposition_cache[char]
            
        # First try the mapping table
        if char in self.char_to_base_diacritic:
            result = self.char_to_base_diacritic[char]
            self.decomposition_cache[char] = result
            return result
        
        # Fall back to Unicode decomposition
        norm_char = unicodedata.normalize('NFD', char)
        
        if len(norm_char) == 1:
            # No diacritics
            result = (char, 'no_diacritic')
        else:
            # Extract base character (first character) and diacritics
            base_char = norm_char[0]
            
            # Identify diacritic type
            if any(c in norm_char for c in '\u0300\u0340'):  # Grave
                diacritic = 'grave'
            elif any(c in norm_char for c in '\u0301\u0341'):  # Acute
                diacritic = 'acute'
            elif any(c in norm_char for c in '\u0303'):  # Tilde
                diacritic = 'tilde'
            elif any(c in norm_char for c in '\u0309'):  # Hook
                diacritic = 'hook'
            elif any(c in norm_char for c in '\u0323'):  # Dot
                diacritic = 'dot'
            elif any(c in norm_char for c in '\u0302'):  # Circumflex
                diacritic = 'circumflex'
            elif any(c in norm_char for c in '\u0306'):  # Breve
                diacritic = 'breve'
            elif any(c in norm_char for c in '\u031b'):  # Horn
                diacritic = 'horn'
            elif any(c in norm_char for c in '\u0327'):  # Stroke
                diacritic = 'stroke'
            else:
                diacritic = 'no_diacritic'
            
            result = (base_char, diacritic)
        
        # Cache the result
        self.decomposition_cache[char] = result
        return result

    def _decompose_text(self, text):
        """Decompose a text string into lists of base characters and diacritics"""
        base_chars = []
        diacritics = []
        
        for char in text:
            base_char, diacritic = self._decompose_vietnamese_char(char)
            base_chars.append(base_char)
            diacritics.append(diacritic)
        
        return base_chars, diacritics