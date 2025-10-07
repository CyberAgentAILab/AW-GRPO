"""
Dataset utilities for machine translation training.

This module provides functions for parsing XML datasets and preparing them for GRPO training.
"""

import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
from datasets import Dataset
import random

def parse_xml_dataset(xml_path: str) -> List[Dict]:
    """
    Parse the XML dataset and extract source and reference translations.
    
    Args:
        xml_path: Path to the XML dataset file
        
    Returns:
        List of dictionaries containing source and reference translations
    """
    print(f"Parsing XML dataset: {xml_path}")
    
    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    data = []
    
    # Iterate through all documents
    for doc in root.findall(".//doc"):
        doc_id = doc.get("id", "")
        
        # Get source segments
        src_segments = {}
        for src in doc.findall("./src[@lang='en']/p/seg"):
            seg_id = src.get("id", "")
            src_text = src.text.strip() if src.text else ""
            src_segments[seg_id] = src_text
        
        # Get reference translations
        ref_segments = {}
        for ref in doc.findall("./ref[@lang='ja']/p/seg"):
            seg_id = ref.get("id", "")
            ref_text = ref.text.strip() if ref.text else ""
            ref_segments[seg_id] = ref_text
        
        # Match source and reference segments
        for seg_id, src_text in src_segments.items():
            if seg_id in ref_segments:
                data.append({
                    "doc_id": doc_id,
                    "seg_id": seg_id,
                    "source": src_text,
                    "reference": ref_segments[seg_id]
                })
    
    print(f"Extracted {len(data)} segment pairs from the dataset")
    return data

def load_datasets(train_paths: List[str], test_path: Optional[str] = None) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Load and prepare datasets for training and testing.
    
    Args:
        train_paths: List of paths to training dataset XML files
        test_path: Path to test dataset XML file (optional)
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Load training data
    train_data = []
    for path in train_paths:
        train_data.extend(parse_xml_dataset(path))
    
    # Create training dataset
    train_dataset = Dataset.from_list(train_data)
    
    # Load test data if provided
    test_dataset = None
    if test_path:
        test_data = parse_xml_dataset(test_path)
        test_dataset = Dataset.from_list(test_data)
    
    return train_dataset, test_dataset

def create_translation_prompt(source_text: str) -> str:
    """
    Create a prompt for the translation task.
    
    Args:
        source_text: Source text to translate
        
    Returns:
        Formatted prompt for the model
    """
    return f"Translate the following English text to Japanese. Do not include texts other than the translation. Translate the input precisely without loss of information.\n\nEnglish text: {source_text}\n\nJapanese translation:"

def prepare_dataset_for_grpo(dataset: Dataset, tokenizer) -> Dataset:
    """
    Prepare dataset for GRPO training by formatting prompts and completions.
    
    Args:
        dataset: Dataset containing source and reference translations
        tokenizer: Tokenizer for the model
        
    Returns:
        Dataset formatted for GRPO training
    """
    def format_example(example):
        prompt = create_translation_prompt(example["source"])
        
        # Format prompt using the chat template
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
        
        return {
            "prompt": formatted_prompt,
            "completion": example["reference"],
            "source": example["source"],
            "ground_truth": example["reference"], # ground_truth is the format for GRPO.
            "doc_id": example["doc_id"],
            "seg_id": example["seg_id"]
        }
    
    # Apply formatting to all examples
    formatted_dataset = dataset.map(format_example)
    
    return formatted_dataset

def split_dataset(dataset: Dataset, test_size: float = 0.1, seed: int = 42) -> Tuple[Dataset, Dataset]:
    """
    Split a dataset into training and validation sets.
    
    Args:
        dataset: Dataset to split
        test_size: Fraction of data to use for validation
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    # Convert to list for easier manipulation
    data = dataset.to_list()
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # Shuffle the data
    random.shuffle(data)
    
    # Calculate split point
    split_idx = int(len(data) * (1 - test_size))
    
    # Split the data
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    # Convert back to Dataset objects
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    
    return train_dataset, val_dataset