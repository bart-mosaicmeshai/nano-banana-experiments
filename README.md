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

### Generate JPEG output with custom path:

```bash
nano-banana generate "blog header image" --output ~/Projects/blog/assets/header.jpg --model 3 --resolution 2K
```

### Control JPEG quality (1-100, default: 85):

```bash
nano-banana generate "optimized web image" --output image.jpg --quality 90 --model 3
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

### Auto-generated Filenames

When no `--output` is specified, images are automatically organized in date-based subdirectories:

```
output/
├── 2024-11-23/
│   ├── generated_v2_1K_20241123_143052.png
│   └── generated_v3_2K_20241123_145321.png
├── 2024-11-24/
│   └── generated_v3_4K_20241124_092145.png
└── generation_log.json
```

### Custom Output Paths

When using `--output`, you can specify:
- **Absolute paths**: `--output ~/Projects/blog/assets/image.jpg` (saves directly to specified location)
- **Relative paths**: `--output my-image.png` (saves to date-based subdirectory)
- **JPEG format**: Automatically converts to JPEG when path ends with `.jpg` or `.jpeg`
- **PNG format**: Saves as PNG when path ends with `.png` or for auto-generated names

The tool automatically:
- Expands `~` to your home directory
- Creates parent directories if they don't exist
- Converts RGBA to RGB for JPEG compatibility

**Note:** JPEG conversion happens locally on your machine after the API call, so there's no additional cost for JPEG output. The API pricing is the same whether you save as PNG or JPEG.

### Generation Logging

All generations are tracked in `output/generation_log.json` with:
- Timestamp
- Prompt text
- Output file path
- Model and resolution used
- Input/reference images (if any)
- Cost breakdown (input, output, and total)

### Image Metadata

Generated images include embedded metadata:

**PNG files** (in PNG info fields):
- Original prompt
- Model name and version
- Resolution setting
- Generation timestamp
- Input/reference image paths (if used)

**JPEG files** (in JPEG comment field as JSON):
- Same metadata as PNG, stored in JSON format
- Optimized for web with configurable quality setting

You can view PNG metadata using image viewers or tools that read PNG metadata. JPEG metadata can be extracted using tools like `exiftool` or by reading the JPEG comment field.

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
