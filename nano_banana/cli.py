#!/usr/bin/env python3
"""CLI tool for Gemini 2.5 Flash Image (nano-banana) model."""

import os
import click
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from google import genai
from PIL import Image

# Load environment variables
load_dotenv()

# Default output directory
OUTPUT_DIR = Path("output")


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
@click.option('--model', '-m', type=click.Choice(['2', '3'], case_sensitive=False), default='2', help='Model version: 2 (nano-banana-2) or 3 (nano-banana-3 Pro)')
@click.option('--resolution', type=click.Choice(['1K', '2K', '4K'], case_sensitive=False), default='1K', help='Output resolution (only for model 3)')
def generate(prompt, image, reference, output, model, resolution):
    """Generate or edit an image based on a text prompt.

    Examples:
        nano-banana generate "a raccoon holding a sign that says I love trash"
        nano-banana generate "add a strawberry to the left eye" -i input.png -o output.png
        nano-banana generate "create a menu for a coffee shop" --model 3 --resolution 2K
        nano-banana generate "person in different scene" -r reference1.jpg -r reference2.jpg --model 3
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        click.echo("Error: GEMINI_API_KEY not found in environment variables.", err=True)
        click.echo("Please set it in your .env file.", err=True)
        return 1

    try:
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Select model based on version
        if model == '3':
            model_name = "gemini-3-pro-image-preview"
            click.echo(f"Using Nano Banana 3 Pro (resolution: {resolution})")
        else:
            model_name = "gemini-2.5-flash-image-preview"
            if resolution != '1K':
                click.echo("Warning: Resolution option ignored for model 2 (always outputs at default resolution)")
            click.echo("Using Nano Banana 2")

        # Generate output filename if not provided
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = OUTPUT_DIR / f"generated_v{model}_{resolution}_{timestamp}.png"
        else:
            output = Path(output)
            # If relative path provided, save to output directory
            if not output.is_absolute():
                output = OUTPUT_DIR / output

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

        # Prepare generation config for model 3
        config = {}
        if model == '3':
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
                img_data.save(output)
                click.echo(f"\nImage saved to: {output}")
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
                if model == '3':
                    # Model 3: $0.0011 per image input
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
                    # Model 3 pricing
                    if resolution == '4K':
                        output_cost = 0.24 * num_images
                    else:  # 1K or 2K
                        output_cost = 0.134 * num_images
                    click.echo(f"Output cost: ${output_cost:.6f} ({num_images} image @ {resolution})")
                else:
                    # Model 2: $0.039 per image (1290 tokens)
                    output_cost = 0.039 * num_images
                    click.echo(f"Output cost: ${output_cost:.6f} ({num_images} image)")

            if hasattr(usage, 'total_token_count'):
                click.echo(f"Total tokens: {usage.total_token_count:,}")

            # Calculate total cost
            total_cost = input_cost + output_cost
            click.echo(f"Total cost: ${total_cost:.6f}")
            click.echo("-------------------")

    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        return 1


@main.command()
def info():
    """Display information about nano-banana capabilities."""
    info_text = """
Nano Banana Models:

═══════════════════════════════════════════════════════════════════════════
MODEL 2: Gemini 2.5 Flash Image (gemini-2.5-flash-image-preview)
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
MODEL 3: Gemini 3 Pro Image (gemini-3-pro-image-preview)
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
