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

### Edit an existing image:

```bash
nano-banana generate "add a strawberry to the left eye" -i input.png -o output.png
```

### Get information about capabilities:

```bash
nano-banana info
```

## API Key

Get your free API key from [Google AI Studio](https://aistudio.google.com/)

## Model Details

- **Model**: gemini-2.5-flash-image-preview
- **Type**: Autoregressive (not diffusion-based)
- **Tokens per image**: 1,290
- **Key advantage**: Exceptional prompt adherence and editing capabilities

## Resources

- [Google AI Studio](https://aistudio.google.com/)
- [Gemini API Documentation](https://ai.google.dev/gemini-api/docs)
