# MBOX to HTML Converter

A Python tool that converts `.mbox` email archives into self-contained HTML viewers with AI-powered image captions.

## ⚠️ Important Warning

**This tool uses the OpenAI API which incurs costs. You use this at your own financial risk.** The tool will estimate costs before processing, but actual charges may vary. Please ensure you understand OpenAI's pricing and have appropriate usage limits set on your account.

## What It Does

This tool solves a specific problem: when you want to share email threads with AI assistants (like ChatGPT) or other tools that need to understand the full context including images. Simply uploading an `.mbox` file often fails because:

- The raw `.mbox` format is difficult to parse
- Embedded and attached images aren't accessible
- Remote images referenced by URL aren't included
- AI assistants can't "see" the images to understand the full context

This converter:
1. **Extracts all emails** from an `.mbox` file
2. **Downloads all remote images** referenced in HTML emails
3. **Extracts all embedded/attached images** including inline CID references
4. **Deduplicates images** using SHA-256 hashing to save space
5. **Generates AI captions** for images using GPT-4o-mini (optional)
6. **Creates a self-contained HTML viewer** with all images included
7. **Packages everything into a `.zip` file** for easy sharing

The result is a complete, portable HTML archive that can be uploaded to ChatGPT or other tools for full context understanding.

## Features

- 📧 Complete email preservation with proper encoding handling
- 🖼️ Automatic image extraction and deduplication
- 🤖 Smart AI captioning with auto-detection of complex images (charts, tables, text-heavy content)
- 💰 Cost estimation before processing
- 🎯 Configurable image size filtering
- 🌓 Dark mode support in the HTML viewer
- 📊 Processing statistics and progress tracking

## Installation

```bash
pip install "openai>=1.3.4" requests beautifulsoup4 pillow tqdm
```

## Usage

```bash
python mbox_to_html.py
```

The interactive wizard will guide you through:
1. Selecting your `.mbox` file
2. Choosing whether to enable AI captions
3. Selecting caption mode (auto-detect, simple, or custom)
4. Setting minimum image size to process
5. Reviewing cost estimates
6. Processing and creating the output

Your OpenAI API key will be requested once and stored locally in `mbox_to_html_keyDONTSHARE.txt`.

## Output

The tool creates:
- 📁 A folder containing:
  - `index.html` - The main viewer
  - `images/` - All extracted and downloaded images
  - `attachments/` - Non-image attachments
  - `captions.json` - AI-generated captions (if enabled)
- 📦 A `.zip` file containing everything above

## Caption Modes

1. **Auto-detect (Recommended)**: Intelligently adapts to image content
   - Simple images: Brief descriptions
   - Complex images (charts, tables, text): Detailed extraction with extended token limits

2. **Simple**: One-sentence factual descriptions only

3. **Custom**: Use your own prompt

## Safety Features

- **Cost estimation** before processing any images
- **Hard limit** of 50 images per run to prevent excessive API costs
- **Automatic deduplication** to avoid processing the same image multiple times
- **Size filtering** to skip small tracking pixels and spacer images

## Use Case

This tool was created to solve the problem of sharing email threads with ChatGPT or other AI assistants. When analyzing email conversations, the context provided by images is often crucial - whether they're screenshots, charts, diagrams, or other visual information. This tool ensures that AI assistants can "see" and understand the complete conversation.

## License

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or
distribute this software, either in source code form or as a compiled
binary, for any purpose, commercial or non-commercial, and by any
means.

In jurisdictions that recognize copyright laws, the author or authors
of this software dedicate any and all copyright interest in the
software to the public domain. We make this dedication for the benefit
of the public at large and to the detriment of our heirs and
successors. We intend this dedication to be an overt act of
relinquishment in perpetuity of all present and future rights to this
software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR
OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <https://unlicense.org/>

## Contributing

This is an active project that will likely be updated. Contributions and suggestions are welcome!
