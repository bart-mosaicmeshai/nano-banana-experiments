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
@click.option('--output', '-o', default=None, help='Output filename (default: auto-generated with timestamp in output/)')
def generate(prompt, image, output):
    """Generate or edit an image based on a text prompt.

    Examples:
        nano-banana generate "a raccoon holding a sign that says I love trash"
        nano-banana generate "add a strawberry to the left eye" -i input.png -o output.png
    """
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        click.echo("Error: GEMINI_API_KEY not found in environment variables.", err=True)
        click.echo("Please set it in your .env file.", err=True)
        return 1

    try:
        # Create output directory if it doesn't exist
        OUTPUT_DIR.mkdir(exist_ok=True)

        # Generate output filename if not provided
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = OUTPUT_DIR / f"generated_{timestamp}.png"
        else:
            output = Path(output)

        # Initialize the client
        client = genai.Client(api_key=api_key)

        # Prepare content
        contents = [prompt]
        if image:
            click.echo(f"Loading image: {image}")
            img = Image.open(image)
            contents.append(img)

        click.echo(f"Generating image with prompt: {prompt}")

        # Generate the image
        response = client.models.generate_content(
            model="gemini-2.5-flash-image-preview",
            contents=contents,
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

            if hasattr(usage, 'prompt_token_count'):
                click.echo(f"Input tokens: {usage.prompt_token_count:,}")
                input_cost = (usage.prompt_token_count / 1_000_000) * 0.30
                click.echo(f"Input cost: ${input_cost:.6f}")

            if hasattr(usage, 'candidates_token_count'):
                click.echo(f"Output tokens: {usage.candidates_token_count:,}")
                # Output images are charged per image, not per token
                # 1290 tokens = $0.039 per image
                num_images = 1 if image_saved else 0
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
Nano Banana (Gemini 2.5 Flash Image) Capabilities:

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

🧠 MULTIMODAL REASONING
   - Understand visual context
   - Follow complex instructions on images

💡 TIPS FOR BEST RESULTS:
   - Be detailed and specific in prompts
   - Avoid buzzwords (hyper-realistic, stunning, etc.)
   - Default style is photorealistic unless specified
   - Can generate and edit trademarked characters

Model: gemini-2.5-flash-image-preview
Tokens per image: 1,290
"""
    click.echo(info_text)


if __name__ == '__main__':
    main()
