"""
ComfyUI Custom Node for Tag Expansion
"""
import os
from pathlib import Path
from .tag_expander import TagExpander

# HuggingFace model repo
MODEL_REPO = "Elldreth/danbooru-tag-implications-flan-t5"

# Singleton instance (load once, reuse)
_expander_instance = None

def get_expander():
    """Get or create the tag expander instance"""
    global _expander_instance
    if _expander_instance is None:
        _expander_instance = TagExpander(MODEL_REPO)
    return _expander_instance

class DanbooruTagExpander:
    """
    Expands Danbooru tags using learned implications
    
    Adds implied tags based on Danbooru tag implication data.
    For example: bikini -> swimsuit, cat_ears -> animal_ears
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "tags": ("STRING", {
                    "multiline": True,
                    "default": "1girl, bikini, cat_ears"
                }),
            },
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("expanded_tags", "added_tags")
    FUNCTION = "expand_tags"
    CATEGORY = "conditioning"
    
    def expand_tags(self, tags):
        """Expand tags with implications"""
        expander = get_expander()
        
        # Handle empty input
        if not tags or not tags.strip():
            return (tags, "")
        
        # Parse input tags
        input_tags = [t.strip() for t in tags.split(',') if t.strip()]
        input_set = set(input_tags)
        
        # Expand tags
        expanded = expander.expand_tags(tags)
        expanded_set = set(t.strip() for t in expanded.split(',') if t.strip())
        
        # Calculate added tags
        added = expanded_set - input_set
        added_str = ', '.join(sorted(added)) if added else ""
        
        return (expanded, added_str)

# Node registration
NODE_CLASS_MAPPINGS = {
    "DanbooruTagExpander": DanbooruTagExpander
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DanbooruTagExpander": "Danbooru Tag Expander"
}
