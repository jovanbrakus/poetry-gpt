import pytest
from data.tokenizer import CharTokenizer


class TestCharTokenizer:
    @pytest.fixture
    def sample_text(self):
        return "abcde"
    
    @pytest.fixture
    def tokenizer(self, sample_text):
        return CharTokenizer(sample_text)
    
    def test_init_creates_vocab(self, tokenizer):
        assert tokenizer.vocab_size == 5
        assert 'a' in tokenizer.char_to_idx
        assert 'e' in tokenizer.char_to_idx
        assert len(tokenizer.char_to_idx) == 5
        assert len(tokenizer.idx_to_char) == 5
    
    def test_encode_basic(self, tokenizer):
        encoded = tokenizer.encode("abc")
        assert len(encoded) == 3
        assert all(isinstance(x, int) for x in encoded)
    
    def test_decode_basic(self, tokenizer):
        original = "abc"
        encoded = tokenizer.encode(original)
        decoded = tokenizer.decode(encoded)
        assert decoded == original
    
    def test_encode_decode_roundtrip(self, tokenizer, sample_text):
        encoded = tokenizer.encode(sample_text)
        decoded = tokenizer.decode(encoded)
        assert decoded == sample_text
    
    def test_encode_unknown_char_raises_keyerror(self, tokenizer):
        with pytest.raises(KeyError):
            tokenizer.encode("z")
    
    def test_decode_invalid_index_raises_keyerror(self, tokenizer):
        with pytest.raises(KeyError):
            tokenizer.decode([999])
    
    def test_empty_string_handling(self, tokenizer):
        assert tokenizer.encode("") == []
        assert tokenizer.decode([]) == ""
    
    def test_vocab_sorted(self):
        text = "dcba"
        tokenizer = CharTokenizer(text)
        chars = list(tokenizer.char_to_idx.keys())
        assert chars == ['a', 'b', 'c', 'd']
    
    def test_duplicate_chars_handled(self):
        text = "aabbcc"
        tokenizer = CharTokenizer(text)
        assert tokenizer.vocab_size == 3
        assert set(tokenizer.char_to_idx.keys()) == {'a', 'b', 'c'}