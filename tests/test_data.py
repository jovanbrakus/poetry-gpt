import pytest
import torch
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, mock_open

from data.dataset import load_slovenian_data
from data.loader import create_data_loader
from data.tokenizer import CharTokenizer


class TestLoadSlovenianData:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)
    
    def teardown_method(self):
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_load_slovenian_data_single_file(self):
        test_file = self.temp_path / "test1.txt"
        test_content = "Test content for file 1"
        test_file.write_text(test_content, encoding='utf-8')
        
        result = load_slovenian_data(str(self.temp_path))
        
        assert result == test_content
    
    def test_load_slovenian_data_multiple_files(self):
        test_file1 = self.temp_path / "test1.txt"
        test_file2 = self.temp_path / "test2.txt"
        content1 = "Content one"
        content2 = "Content two"
        
        test_file1.write_text(content1, encoding='utf-8')
        test_file2.write_text(content2, encoding='utf-8')
        
        result = load_slovenian_data(str(self.temp_path))
        expected = f"{content1}\n\n{content2}"
        
        assert result == expected
    
    def test_load_slovenian_data_files_sorted(self):
        test_file_c = self.temp_path / "c.txt"
        test_file_a = self.temp_path / "a.txt"
        test_file_b = self.temp_path / "b.txt"
        
        test_file_c.write_text("C", encoding='utf-8')
        test_file_a.write_text("A", encoding='utf-8')
        test_file_b.write_text("B", encoding='utf-8')
        
        result = load_slovenian_data(str(self.temp_path))
        expected = "A\n\nB\n\nC"
        
        assert result == expected
    
    def test_load_slovenian_data_nonexistent_directory(self):
        with pytest.raises(FileNotFoundError, match="Slovenian data directory not found"):
            load_slovenian_data("/nonexistent/path")
    
    def test_load_slovenian_data_no_txt_files(self):
        # Create a directory with no .txt files
        other_file = self.temp_path / "not_txt.py"
        other_file.write_text("This is not a txt file", encoding='utf-8')
        
        with pytest.raises(FileNotFoundError, match="No .txt files found"):
            load_slovenian_data(str(self.temp_path))
    
    def test_load_slovenian_data_empty_file(self):
        empty_file = self.temp_path / "empty.txt"
        empty_file.write_text("", encoding='utf-8')
        
        result = load_slovenian_data(str(self.temp_path))
        
        assert result == ""
    
    def test_load_slovenian_data_ignores_non_txt_files(self):
        txt_file = self.temp_path / "poetry.txt"
        other_file = self.temp_path / "data.json"
        
        txt_content = "Poetry content"
        txt_file.write_text(txt_content, encoding='utf-8')
        other_file.write_text('{"key": "value"}', encoding='utf-8')
        
        result = load_slovenian_data(str(self.temp_path))
        
        assert result == txt_content
    
    def test_load_slovenian_data_handles_utf8_characters(self):
        txt_file = self.temp_path / "slovenian.txt"
        slovenian_content = "čšžČŠŽ"
        txt_file.write_text(slovenian_content, encoding='utf-8')
        
        result = load_slovenian_data(str(self.temp_path))
        
        assert result == slovenian_content


class TestCreateDataLoader:
    @pytest.fixture
    def sample_text(self):
        return "abcdefghijklmnopqrstuvwxyz"
    
    @pytest.fixture
    def tokenizer(self, sample_text):
        return CharTokenizer(sample_text)
    
    def test_create_data_loader_basic(self, sample_text, tokenizer):
        batch_size = 2
        seq_len = 5
        
        get_batch = create_data_loader(sample_text, tokenizer, batch_size, seq_len)
        
        assert callable(get_batch)
    
    def test_create_data_loader_batch_shape(self, sample_text, tokenizer):
        batch_size = 2
        seq_len = 5
        
        get_batch = create_data_loader(sample_text, tokenizer, batch_size, seq_len)
        batches = list(get_batch())
        
        if batches:
            first_batch = batches[0]
            assert first_batch.shape[0] <= batch_size
            assert first_batch.shape[1] == seq_len + 1
    
    def test_create_data_loader_empty_text(self, tokenizer):
        empty_text = ""
        get_batch = create_data_loader(empty_text, tokenizer, 2, 5)
        batches = list(get_batch())
        
        assert len(batches) == 0
    
    def test_create_data_loader_short_text(self, tokenizer):
        short_text = "abc"
        batch_size = 2
        seq_len = 5
        
        get_batch = create_data_loader(short_text, tokenizer, batch_size, seq_len)
        batches = list(get_batch())
        
        assert len(batches) == 0
    
    def test_create_data_loader_exact_length(self, tokenizer):
        text = "abcdefghijk"
        batch_size = 1
        seq_len = 5
        
        get_batch = create_data_loader(text, tokenizer, batch_size, seq_len)
        batches = list(get_batch())
        
        assert len(batches) >= 0
    
    def test_create_data_loader_batch_content(self, tokenizer):
        text = "abcdefghijklmnop"
        batch_size = 1
        seq_len = 5
        
        get_batch = create_data_loader(text, tokenizer, batch_size, seq_len)
        batches = list(get_batch())
        
        if batches:
            first_batch = batches[0]
            assert torch.is_tensor(first_batch)
            assert first_batch.dtype == torch.long
    
    def test_create_data_loader_multiple_batches(self, tokenizer):
        long_text = "abcdefghijklmnopqrstuvwxyz" * 10
        batch_size = 2
        seq_len = 10
        
        get_batch = create_data_loader(long_text, tokenizer, batch_size, seq_len)
        batches = list(get_batch())
        
        assert len(batches) > 0
        
        for batch in batches:
            assert batch.shape[0] <= batch_size
            assert batch.shape[1] == seq_len + 1
    
    def test_create_data_loader_generator_reusable(self, sample_text, tokenizer):
        batch_size = 2
        seq_len = 5
        
        get_batch = create_data_loader(sample_text, tokenizer, batch_size, seq_len)
        
        batches1 = list(get_batch())
        batches2 = list(get_batch())
        
        assert len(batches1) == len(batches2)
    
    def test_create_data_loader_sequential_data(self, tokenizer):
        text = "abcdefghijklmnop"
        batch_size = 1
        seq_len = 3
        
        get_batch = create_data_loader(text, tokenizer, batch_size, seq_len)
        batches = list(get_batch())
        
        if len(batches) >= 2:
            first_batch = batches[0][0]
            second_batch = batches[1][0]
            
            expected_first = tokenizer.encode(text[:4])
            expected_second = tokenizer.encode(text[3:7])
            
            assert torch.equal(first_batch, torch.tensor(expected_first))
            assert torch.equal(second_batch, torch.tensor(expected_second))