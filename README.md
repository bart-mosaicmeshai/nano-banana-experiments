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
3. Install dependencies:

```bash
pip install -e .
```

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

### Get information about capabilities:

```bash
nano-banana info
```

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
