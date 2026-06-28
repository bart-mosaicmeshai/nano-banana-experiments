#!/usr/bin/env python3
"""CLI tool for Gemini 2.5 Flash Image (nano-banana) model."""

import os
import json
import click
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Load environment variables
load_dotenv()

# Default output directory
OUTPUT_DIR = Path("output")

# Log file for generation history
LOG_FILE = OUTPUT_DIR / "generation_log.json"


def log_generation(prompt, output_path, model, resolution, image_path=None, reference_paths=None, cost_info=None):
    """Log generation details to JSON file."""
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Load existing log or create new one
    log_entries = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE, 'r') as f:
                log_entries = json.load(f)
        except json.JSONDecodeError:
            log_entries = []

    # Create new entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "output_file": str(output_path),
        "model": model,
        "resolution": resolution,
        "input_image": str(image_path) if image_path else None,
        "reference_images": [str(p) for p in reference_paths] if reference_paths else [],
    }

    # Add cost information if available
    if cost_info:
        entry["cost"] = cost_info

    log_entries.append(entry)

    # Save updated log
    with open(LOG_FILE, 'w') as f:
        json.dump(log_entries, f, indent=2)

    return entry


@click.group()
@click.version_option()
def main():
    """Nano Banana CLI - Explore Google's Gemini 2.5 Flash Image model."""
    pass


@main.command()
@click.argument('prompt')
@click.option('--image', '-i', type=click.Path(exists=True), help='Input image for editing/composition')
@click.option('--reference', '-r', multiple=True, type=click.Path(exists=True), help='Reference images for consistency (can be used multiple times)')
@click.option('--output', '-o', default=None, help='Output filename (default: auto-generated with timestamp in output/)')
@click.option('--model', '-m', type=click.Choice(['2', '3', '3.1'], case_sensitive=False), default='2', help='Model: 2 (Gemini 2.5 Flash Image), 3.1 (Gemini 3.1 Flash Image), or 3 (Gemini 3 Pro Image)')
@click.option('--resolution', type=click.Choice(['1K', '2K', '4K'], case_sensitive=False), default='1K', help='Output resolution (only for model 3)')
@click.option('--quality', '-q', type=int, default=85, help='JPEG quality (1-100, default: 85, only applies to JPEG output)')
def generate(prompt, image, reference, output, model, resolution, quality):
    """Generate or edit an image based on a text prompt.

    Examples:
        nano-banana generate "a raccoon holding a sign that says I love trash"
        nano-banana generate "add a strawberry to the left eye" -i input.png -o output.png
        nano-banana generate "create a menu for a coffee shop" --model 3 --resolution 2K
        nano-banana generate "person in different scene" -r reference1.jpg -r reference2.jpg --model 3
        nano-banana generate "blog header image" --output ~/Projects/blog/assets/header.jpg --quality 90
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        click.echo("Error: GEMINI_API_KEY not found in environment variables.", err=True)
        click.echo("Please set it in your .env file.", err=True)
        return 1

    try:
        # Select model based on version (GA model IDs as of 2026)
        if model == '3':
            model_name = "gemini-3-pro-image"
            click.echo(f"Using Gemini 3 Pro Image / Nano Banana Pro (resolution: {resolution})")
        elif model == '3.1':
            model_name = "gemini-3.1-flash-image"
            click.echo(f"Using Gemini 3.1 Flash Image / Nano Banana 2 (resolution: {resolution})")
        else:
            model_name = "gemini-2.5-flash-image"
            if resolution != '1K':
                click.echo("Warning: Resolution option ignored for model 2 (always outputs at default resolution)")
            click.echo("Using Gemini 2.5 Flash Image / Nano Banana")

        # Handle output path
        use_date_dir = False
        if output is None:
            # Auto-generated filename: use date-based subdirectory
            use_date_dir = True
            today = datetime.now().strftime("%Y-%m-%d")
            date_dir = OUTPUT_DIR / today
            date_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = date_dir / f"generated_v{model}_{resolution}_{timestamp}.png"
        else:
            # User-specified output path
            output = Path(output).expanduser()  # Expand ~ to home directory

            # If relative path provided, save to date directory
            if not output.is_absolute():
                use_date_dir = True
                today = datetime.now().strftime("%Y-%m-%d")
                date_dir = OUTPUT_DIR / today
                date_dir.mkdir(parents=True, exist_ok=True)
                output = date_dir / output
            else:
                # Absolute path: create parent directories if needed
                output.parent.mkdir(parents=True, exist_ok=True)

        # Determine output format based on extension
        output_format = output.suffix.lower()
        is_jpeg = output_format in ['.jpg', '.jpeg']

        # Initialize the client
        client = genai.Client(api_key=api_key)

        # Prepare content
        contents = [prompt]

        # Add reference images if provided
        if reference:
            for ref_path in reference:
                click.echo(f"Loading reference image: {ref_path}")
                ref_img = Image.open(ref_path)
                contents.append(ref_img)

        # Add input image if provided
        if image:
            click.echo(f"Loading input image: {image}")
            img = Image.open(image)
            contents.append(img)

        click.echo(f"Generating image with prompt: {prompt}")

        # Prepare generation config for model 3 / 3.1
        config = {}
        if model in ('3', '3.1'):
            # Map resolution to generation config
            resolution_map = {
                '1K': {'response_modalities': ['IMAGE']},
                '2K': {'response_modalities': ['IMAGE']},
                '4K': {'response_modalities': ['IMAGE']},
            }
            config = resolution_map.get(resolution, {})

        # Generate the image
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config if config else None,
        )

        # Process the response
        image_saved = False
        for part in response.parts:
            if part.text:
                click.echo(f"\nResponse: {part.text}")
            elif img_data := part.as_image():
                # Prepare metadata
                timestamp_str = datetime.now().isoformat()
                metadata_dict = {
                    "prompt": prompt,
                    "model": model_name,
                    "resolution": resolution,
                    "timestamp": timestamp_str,
                }
                if image:
                    metadata_dict["input_image"] = str(image)
                if reference:
                    metadata_dict["reference_images"] = ", ".join(str(r) for r in reference)

                if is_jpeg:
                    # For JPEG output, save as temp PNG first, then convert
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        tmp_path = tmp.name

                    # Save the image from API as PNG
                    img_data.save(tmp_path)

                    # Re-open with PIL to convert to JPEG
                    pil_image = Image.open(tmp_path)

                    # Convert RGBA to RGB if needed (JPEG doesn't support alpha)
                    if pil_image.mode in ('RGBA', 'LA', 'P'):
                        # Create white background
                        background = Image.new('RGB', pil_image.size, (255, 255, 255))
                        if pil_image.mode == 'P':
                            pil_image = pil_image.convert('RGBA')
                        background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode in ('RGBA', 'LA') else None)
                        pil_image = background
                    elif pil_image.mode != 'RGB':
                        pil_image = pil_image.convert('RGB')

                    # Save as JPEG with metadata in comment
                    metadata_json = json.dumps(metadata_dict)
                    pil_image.save(output, format='JPEG', quality=quality, optimize=True,
                                 comment=metadata_json)

                    # Clean up temp file
                    Path(tmp_path).unlink()

                    click.echo(f"\nJPEG image saved to: {output} (quality: {quality})")
                else:
                    # Save as PNG with metadata
                    # Save the image first
                    img_data.save(output)

                    # Re-open with PIL to add metadata
                    pil_image = Image.open(output)

                    # Create PNG metadata with prompt information
                    png_metadata = PngInfo()
                    for key, value in metadata_dict.items():
                        png_metadata.add_text(key, value)

                    # Save again with metadata
                    pil_image.save(output, pnginfo=png_metadata)
                    click.echo(f"\nPNG image saved to: {output}")

                image_saved = True

        if not image_saved:
            click.echo("Warning: No image was generated in the response.", err=True)
            return 1

        # Display usage and cost information
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            click.echo("\n--- Usage & Cost ---")
            click.echo(f"Model: {model_name}")

            input_cost = 0.0
            output_cost = 0.0

            if hasattr(usage, 'prompt_token_count'):
                click.echo(f"Input tokens: {usage.prompt_token_count:,}")

                # Calculate input cost based on model
                if model in ('3', '3.1'):
                    # Model 3 / 3.1: ~$0.0011 per input image
                    num_input_images = len(reference) + (1 if image else 0)
                    input_cost = num_input_images * 0.0011
                    if num_input_images > 0:
                        click.echo(f"Input images: {num_input_images}")
                        click.echo(f"Input cost: ${input_cost:.6f}")
                else:
                    # Model 2: $0.30 per million input tokens
                    input_cost = (usage.prompt_token_count / 1_000_000) * 0.30
                    click.echo(f"Input cost: ${input_cost:.6f}")

            if hasattr(usage, 'candidates_token_count'):
                click.echo(f"Output tokens: {usage.candidates_token_count:,}")

                # Calculate output cost based on model and resolution
                num_images = 1 if image_saved else 0
                if model == '3':
                    # Gemini 3 Pro Image: $0.134 (1K/2K), $0.24 (4K)
                    output_cost = (0.24 if resolution == '4K' else 0.134) * num_images
                    click.echo(f"Output cost: ${output_cost:.6f} ({num_images} image @ {resolution})")
                elif model == '3.1':
                    # Gemini 3.1 Flash Image: 0.5K $0.045, 1K $0.067, 2K $0.101, 4K $0.151
                    flash31_pricing = {'0.5K': 0.045, '1K': 0.067, '2K': 0.101, '4K': 0.151}
                    output_cost = flash31_pricing.get(resolution, 0.067) * num_images
                    click.echo(f"Output cost: ${output_cost:.6f} ({num_images} image @ {resolution})")
                else:
                    # Gemini 2.5 Flash Image: $0.039 per image (1290 tokens)
                    output_cost = 0.039 * num_images
                    click.echo(f"Output cost: ${output_cost:.6f} ({num_images} image)")

            if hasattr(usage, 'total_token_count'):
                click.echo(f"Total tokens: {usage.total_token_count:,}")

            # Calculate total cost
            total_cost = input_cost + output_cost
            click.echo(f"Total cost: ${total_cost:.6f}")
            click.echo("-------------------")

            # Log generation to JSON file
            cost_info = {
                "input_cost": input_cost,
                "output_cost": output_cost,
                "total_cost": total_cost,
                "input_tokens": usage.prompt_token_count if hasattr(usage, 'prompt_token_count') else None,
                "output_tokens": usage.candidates_token_count if hasattr(usage, 'candidates_token_count') else None,
            }
            log_generation(prompt, output, model_name, resolution, image, reference, cost_info)
        else:
            # Log even without cost information
            log_generation(prompt, output, model_name, resolution, image, reference)

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@main.command()
def info():
    """Display information about nano-banana capabilities."""
    info_text = """
Nano Banana Models:

═══════════════════════════════════════════════════════════════════════════
MODEL 2: Gemini 2.5 Flash Image (gemini-2.5-flash-image)
═══════════════════════════════════════════════════════════════════════════

🎨 IMAGE GENERATION
   - Create images from text prompts
   - Native image generation (autoregressive, not diffusion)
   - Exceptional prompt adherence

✏️  IMAGE EDITING
   - Inpainting: Add or change objects
   - Outpainting: Extend images beyond borders
   - Targeted transformations

🎭 CHARACTER CONSISTENCY
   - Maintain subject appearance across multiple scenes
   - Preserve visual identity

🖼️  IMAGE COMPOSITION
   - Merge elements from multiple images
   - Create photorealistic composites

💰 PRICING:
   - Input: $0.30 per 1M tokens
   - Output: $0.039 per image (1,290 tokens)

═══════════════════════════════════════════════════════════════════════════
MODEL 3.1: Gemini 3.1 Flash Image (gemini-3.1-flash-image)  [Nano Banana 2]
═══════════════════════════════════════════════════════════════════════════

🍌 NEWER FLASH MODEL (released Feb 2026):
   - Improved quality and character consistency over Model 2
   - Resolutions: 0.5K, 1K, 2K, 4K
   - Good balance of quality and cost between Model 2 and Model 3

💰 PRICING:
   - Output: $0.045 (0.5K), $0.067 (1K), $0.101 (2K), $0.151 (4K)

═══════════════════════════════════════════════════════════════════════════
MODEL 3: Gemini 3 Pro Image (gemini-3-pro-image)
═══════════════════════════════════════════════════════════════════════════

✨ NEW CAPABILITIES:
   - Higher resolutions: 1K, 2K, and 4K output
   - Legible text rendering for infographics, menus, diagrams
   - Multi-reference images: Up to 14 images (6 objects + 5 humans)
   - Thinking mode: Intermediate draft images before final output
   - Google Search integration for fact verification
   - Multi-character editing and consistency
   - Chart and diagram generation
   - Doodle editing support

📏 RESOLUTIONS:
   - 1K: Suitable for web and social media
   - 2K: High-quality prints and detailed work
   - 4K: Professional-grade, ultra-high resolution

💰 PRICING:
   - Input: $0.0011 per image
   - Output 1K/2K: $0.134 per image
   - Output 4K: $0.24 per image

🔐 SECURITY:
   - All images include imperceptible SynthID watermark

═══════════════════════════════════════════════════════════════════════════
💡 TIPS FOR BEST RESULTS:
═══════════════════════════════════════════════════════════════════════════
   - Be detailed and specific in prompts
   - Avoid buzzwords (hyper-realistic, stunning, etc.)
   - Default style is photorealistic unless specified
   - Use reference images for character consistency (Model 3)
   - Specify text content explicitly for infographics (Model 3)
"""
    click.echo(info_text)


if __name__ == '__main__':
    main()
