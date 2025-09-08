class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text):
        """Convert text to indices"""
        return [self.char_to_idx[ch] for ch in text]

    def decode(self, indices):
        """Convert indices back to text"""
        return ''.join([self.idx_to_char[idx] for idx in indices])