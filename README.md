# Nano Banana Experiments

Exploring Google's Gemini 2.5 Flash Image (nano-banana) model - a state-of-the-art autoregressive image generation and editing model.

## Features

- Generate images from text prompts
- Edit existing images with natural language instructions
- Character consistency across multiple generations
- Image composition and merging
- Multimodal reasoning capabilities

## Setup

1. Clone the repository
2. Copy `.env.example` to `.env` and add your Gemini API key
3. Create and activate virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install dependencies:

```bash
pip install -e .
```

**Note:** You'll need to activate the virtual environment (`source venv/bin/activate`) each time you open a new terminal session.

## Usage

### Generate an image from text:

```bash
nano-banana generate "a raccoon holding a sign that says I love trash"
```

### Generate with Model 3 (higher resolution, text rendering):

```bash
nano-banana generate "create a menu for a coffee shop" --model 3 --resolution 2K
```

### Edit an existing image:

```bash
nano-banana generate "add a strawberry to the left eye" -i input.png -o output.png
```

### Use reference images for character consistency (Model 3):

```bash
nano-banana generate "person in different scene" -r reference1.jpg -r reference2.jpg --model 3
```

### Get detailed information about model capabilities:

```bash
nano-banana info
```

This displays comprehensive information about both models including:
- Image generation and editing capabilities
- Resolution options
- Pricing details
- Tips for best results

## Output Organization

Generated images are automatically organized in date-based subdirectories:

```
output/
├── 2024-11-23/
│   ├── generated_v2_1K_20241123_143052.png
│   └── generated_v3_2K_20241123_145321.png
├── 2024-11-24/
│   └── generated_v3_4K_20241124_092145.png
└── generation_log.json
```

### Generation Logging

All generations are tracked in `output/generation_log.json` with:
- Timestamp
- Prompt text
- Output file path
- Model and resolution used
- Input/reference images (if any)
- Cost breakdown (input, output, and total)

### Image Metadata

Each generated PNG file includes embedded metadata:
- Original prompt
- Model name and version
- Resolution setting
- Generation timestamp
- Input/reference image paths (if used)

You can view this metadata using image viewers or tools that read PNG metadata.

## API Key

Get your free API key from [Google AI Studio](https://aistudio.google.com/)

## Model Details

### Model 2 (default): gemini-2.5-flash-image-preview
- **Type**: Autoregressive (not diffusion-based)
- **Tokens per image**: 1,290
- **Key advantage**: Exceptional prompt adherence and editing capabilities
- **Pricing**: $0.039 per image

### Model 3: gemini-3-pro-image-preview
- **New capabilities**: Higher resolutions (1K/2K/4K), legible text rendering, multi-reference images
- **Use cases**: Infographics, menus, diagrams, charts, multi-character scenes
- **Pricing**: $0.134 per image (1K/2K) or $0.24 per image (4K)
- **Security**: Includes imperceptible SynthID watermark

## Resources

- [Google AI Studio](https://aistudio.google.com/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
