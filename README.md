# Danbooru Tag Expander - ComfyUI Custom Node

Automatically expands tags with their implied tags based on official Danbooru tag implication data. Works with both Danbooru-style underscored tags (`cat_ears`) and natural space-separated tags (`cat ears`).

## What It Does

Adds implied tags to your prompts based on Danbooru tag relationships:

- `bikini`  adds `swimsuit`
- `cat_ears`  adds `animal_ears`
- `striped_panties`  adds `panties, striped_clothes`
- `sleeveless_dress`  adds `dress, sleeveless`

## Features

-  **35,000+ tag implications** from Danbooru
-  **Fast inference** (~200ms per tag with implications)
-  **No hallucinations** - only expands known tags
-  **No series-specific tags** - generic tags stay generic
-  **Format auto-detection** - handles both `cat_ears` and `cat ears`
-  **Lazy loading** - model loads once and stays in memory
-  **GPU accelerated** (CUDA) with CPU fallback

## Installation

### 1. Copy to ComfyUI

Copy the `comfy_tag_expander` folder to your ComfyUI custom nodes directory:

```bash
# Windows
xcopy /E /I comfy_tag_expander "C:\ComfyUI\custom_nodes\comfy_tag_expander"

# Linux/Mac
cp -r comfy_tag_expander /path/to/ComfyUI/custom_nodes/
```

### 2. Install Dependencies

The node requires:
- `transformers`
- `torch`
- `huggingface_hub`

These should already be installed in ComfyUI. If not:

```bash
# In ComfyUI's python environment
pip install transformers torch huggingface_hub
```

### 3. Restart ComfyUI

Restart ComfyUI and the node will appear in the **Add Node → conditioning** menu as **"Danbooru Tag Expander"**.

**First Run:** The model (~945MB) will auto-download from HuggingFace to your cache directory (`~/.cache/huggingface/`). This takes 1-2 minutes depending on your connection. Subsequent runs use the cached model.

## Usage

1. **Add the node** to your workflow
2. **Input your tags** (comma-separated)
3. **Connect the output** to your CLIP Text Encode node

### Example

**Input:**
```
1girl, bikini, cat_ears, sleeveless_dress
```

**Output:**
```
1girl, animal_ears, bikini, cat_ears, dress, sleeveless, sleeveless_dress, swimsuit
```

**Added tags:**
- `animal_ears` (from `cat_ears`)
- `swimsuit` (from `bikini`)
- `dress, sleeveless` (from `sleeveless_dress`)

### Format Handling

The node automatically detects and preserves your tag format:

**Underscore format (Danbooru style):**
- Input: `1girl, cat_ears, striped_panties`
- Output: `1girl, animal_ears, cat_ears, panties, striped_clothes, striped_panties`

**Space format (Natural prompting):**
- Input: `1girl, cat ears, striped panties`
- Output: `1girl, animal ears, cat ears, panties, striped clothes, striped panties`

**Mixed format:** If ANY tag uses underscores, output uses underscores for consistency.

## How It Works

1. Takes each input tag
2. Queries the model: "what does this tag imply?"
3. Adds all implications to the tag set
4. Returns sorted, expanded tags

**Guard mechanism:** Only queries the model for tags that exist in the Danbooru implications database. Unknown tags are left unchanged (no hallucinations).

## Performance

- **First run:** ~2-3 seconds (model loading)
- **Subsequent runs:** ~200ms per tag with implications
- **Tags without implications:** 0ms (skipped)
- **Typical 10-tag prompt:** ~1-2 seconds

## Limitations

- **Format flexible but tag-specific** - Works with both `cat_ears` and `cat ears`, but must be actual Danbooru tag names (not natural language descriptions)
- **Generic tags stay generic** - Won't suggest series-specific tags (e.g., `beach` won't suggest `beach_(naruto)`)
- **Only known tags expand** - Model only adds implications for tags in the Danbooru database (32k+ tags)
- **Requires ~1.5GB VRAM** when loaded on GPU

## Troubleshooting

**Node doesn't appear:**
- Check ComfyUI console for errors
- Verify folder structure matches: `custom_nodes/comfy_tag_expander/`
- Ensure `__init__.py` exists

**Model not found error:**
- Check `MODEL_PATH` in `nodes.py` points to correct location
- Verify `tag_implications_model` folder contains model files

**Slow inference:**
- Model loads on first use (normal)
- Ensure CUDA is available for GPU acceleration
- Check GPU memory isn't full

**Unexpected results:**
- Node only expands tags that have implications in Danbooru data
- Not all tags have implications (e.g., `1girl`, `beach`, `thighhighs`)

## Development

Built with:
- FLAN-T5 Base (248M parameters)
- Trained on 32,331 Danbooru tag implication pairs
- 3 epochs, ~30 minutes training time
- BF16 precision, Seq2SeqTrainer

## License

Model trained on Danbooru tag implication data (public dataset).
