"""
Core tag expansion logic using trained implications model
"""
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import hf_hub_download
import json
from pathlib import Path
from typing import Set, List

class TagExpander:
    """Expands tags using Danbooru tag implications model"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path  # Can be HF repo or local path
        self.model = None
        self.tokenizer = None
        self.device = None
        self.tags_with_implications: Set[str] = set()
        
    def load_model(self):
        """Lazy load the model and implications list"""
        if self.model is not None:
            return  # Already loaded
        
        print(f"[TagExpander] Loading model from {self.model_path}...")
        
        # Load from HuggingFace or local path - transformers handles both
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path)
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        
        # Load tags that have implications (to avoid hallucinations)
        # For HF repos, download the dataset file
        try:
            if "/" in self.model_path:  # Likely a HF repo (contains username/repo)
                print("[TagExpander] Downloading implications dataset from HuggingFace...")
                implications_file = hf_hub_download(
                    repo_id=self.model_path,
                    filename="tag_implications_dataset.jsonl",
                    repo_type="model"
                )
            else:  # Local path
                implications_file = Path(self.model_path).parent / "tag_implications_dataset.jsonl"
            
            with open(implications_file, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    tag = data['input'].replace('implications: ', '')
                    self.tags_with_implications.add(tag)
            
            print(f"[TagExpander] Loaded {len(self.tags_with_implications):,} tags with implications")
        except Exception as e:
            print(f"[TagExpander] Warning: Could not load implications dataset: {e}")
            print("[TagExpander] Model will work but may hallucinate on unknown tags")
        
        print(f"[TagExpander] Model loaded on {self.device}")
    
    def get_implications(self, tag: str, use_underscores: bool = True) -> List[str]:
        """Get implications for a single tag
        
        Args:
            tag: Tag to get implications for
            use_underscores: Whether to return tags with underscores or spaces
        """
        self.load_model()  # Lazy load
        
        tag = tag.strip()
        
        # Normalize to underscore format for lookup (model trained on underscored tags)
        normalized_tag = tag.replace(' ', '_')
        
        # Guard: only query model for known tags
        if normalized_tag not in self.tags_with_implications:
            return []
        
        input_text = f"implications: {normalized_tag}"
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                num_beams=4,
            )
        
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        implied_tags = [t.strip() for t in result.split(',') if t.strip()]
        
        # Convert to requested format
        if not use_underscores:
            implied_tags = [t.replace('_', ' ') for t in implied_tags]
        
        return implied_tags
    
    def expand_tags(self, tags_string: str) -> str:
        """Expand all tags in a comma-separated string
        
        Auto-detects format (underscores vs spaces) from input and maintains it in output.
        """
        input_tags = [t.strip() for t in tags_string.split(',') if t.strip()]
        
        if not input_tags:
            return tags_string
        
        # Detect format: if ANY tag uses underscores, use underscore format
        # (Danbooru tags often mix, e.g. "1girl, cat_ears")
        use_underscores = any('_' in tag for tag in input_tags)
        
        # Normalize input tags to the detected format
        if use_underscores:
            # Keep underscores as-is
            normalized_input = input_tags
        else:
            # Convert spaces to underscores for processing, will convert back later
            normalized_input = [tag.replace(' ', '_') for tag in input_tags]
        
        expanded = set(normalized_input)
        
        for tag in normalized_input:
            implied = self.get_implications(tag, use_underscores=True)  # Always get underscored
            expanded.update(implied)
        
        # Convert output to match input format
        if not use_underscores:
            expanded = {tag.replace('_', ' ') for tag in expanded}
        
        return ', '.join(sorted(expanded))
    
    def unload_model(self):
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.tokenizer = None
            print("[TagExpander] Model unloaded")
