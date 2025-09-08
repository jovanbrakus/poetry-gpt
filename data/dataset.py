import os
from pathlib import Path


def load_slovenian_data(data_path="static/slovenian"):
    """Load and concatenate all Slovenian poetry text files"""
    text_data = []
    
    slovenian_path = Path(data_path)
    
    if not slovenian_path.exists():
        raise FileNotFoundError(f"Slovenian data directory not found: {data_path}")
    
    txt_files = list(slovenian_path.glob("*.txt"))
    
    if not txt_files:
        raise FileNotFoundError(f"No .txt files found in {data_path}")
    
    for txt_file in sorted(txt_files):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                content = f.read()
                text_data.append(content)
                print(f"Loaded: {txt_file.name} ({len(content):,} characters)")
        except Exception as e:
            print(f"Error loading {txt_file.name}: {e}")
            continue
    
    combined_text = '\n\n'.join(text_data)
    print(f"\nTotal loaded: {len(txt_files)} files, {len(combined_text):,} characters")
    
    return combined_text