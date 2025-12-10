# Nano Banana Enhancement: Automated JPEG Output

## Context

I'm working on a blog series that requires generating images for each post. The current workflow has manual friction that needs to be eliminated.

**Current workflow (manual):**
1. Run `nano-banana generate "prompt" --model 3 --resolution 1K`
2. PNG is saved to `output/YYYY-MM-DD/generated_v3_1K_timestamp.png`
3. Manually convert PNG → JPEG
4. Manually resize/optimize file size
5. Manually move to target location: `~/Projects/mosaic-mesh-ai-blog/assets/series-name/image-name.jpg`

**Desired workflow (automated):**
1. Run `nano-banana generate "prompt" --output ~/Projects/mosaic-mesh-ai-blog/assets/series-name/image-name.jpg --model 3 --resolution 1K`
2. Tool automatically handles everything: generation, PNG→JPEG conversion, size optimization, saves to specified path

## Task

Enhance the nano-banana CLI to support direct JPEG output with the following requirements:

### 1. Add JPEG Output Support
- When `--output` path ends with `.jpg` or `.jpeg`, automatically convert PNG to JPEG
- Maintain existing PNG workflow when output ends with `.png` or uses auto-generated names
- Add `--quality` option (default: 85) for JPEG compression control

### 2. File Size Optimization
- Optimize JPEG file size for web usage (target: reasonable quality at smaller file size)
- Maintain image metadata in JPEG files (prompt, model, resolution, timestamp)

### 3. Output Path Handling
- Support absolute paths (e.g., `~/Projects/path/to/image.jpg`)
- Expand `~` to user home directory
- Create parent directories if they don't exist
- When output path is specified, save directly there (don't use date-based subdirectories)
- Keep existing auto-generated filename behavior when `--output` is not specified

### 4. Backward Compatibility
- Don't break existing PNG workflow
- Keep date-based subdirectory structure for auto-generated files
- Keep generation logging in `generation_log.json`

## Technical Details

**Files to modify:**
- `nano_banana/cli.py` - Main CLI logic (generate command)

**Python libraries available:**
- PIL/Pillow (already used) - has JPEG conversion and quality settings
- pathlib (already used) - for path handling
- All existing dependencies

**Key implementation points:**
1. Detect output extension to determine format
2. Use `PIL.Image.save(path, format='JPEG', quality=85, optimize=True)` for JPEG conversion
3. Preserve metadata using JPEG EXIF/comment fields (PIL supports this)
4. Expand `~` using `Path.expanduser()`
5. Create parent dirs with `Path.mkdir(parents=True, exist_ok=True)`

## Testing

After implementation, test with:
```bash
# Test JPEG output with explicit path
nano-banana generate "a simple test diagram" --output ~/Desktop/test-image.jpg --model 3 --resolution 1K

# Test PNG still works
nano-banana generate "a simple test diagram" --output ~/Desktop/test-image.png --model 3 --resolution 1K

# Test auto-generated filename still works (backward compatibility)
nano-banana generate "a simple test diagram" --model 3 --resolution 1K

# Test quality option
nano-banana generate "a simple test diagram" --output ~/Desktop/test-low-quality.jpg --quality 60 --model 3 --resolution 1K
```

## Success Criteria

1. Can generate JPEG directly with `--output path/to/file.jpg`
2. JPEG files are optimized for web (reasonable size)
3. Metadata is preserved in JPEG files
4. PNG workflow unchanged
5. Auto-generated filenames still work
6. All existing tests pass

## Context for Blog Post

After this enhancement is complete, I'll return to my blog series work and use this new functionality to generate images for Parts 7-9. Then I'll write a blog post about this enhancement showing:
- The problem (manual workflow friction)
- The solution (automated conversion)
- Real usage examples from the blog series
- Code walkthrough
- Impact on daily workflow efficiency

**Related files:**
- Enhancement notes: `~/Projects/mosaic-mesh-ai-blog/personal-notes/nano-banana-enhancements.md`
- Session plan: `~/Projects/mosaic-mesh-ai-blog/personal-notes/agentic-personal-trainer-series-plan.md`

---

Ready to implement this enhancement. Let's make blog image generation friction-free!
