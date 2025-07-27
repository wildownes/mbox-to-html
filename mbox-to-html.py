#!/usr/bin/env python3
"""
mbox_to_html.py – v4.0
Interactive wizard to convert an .mbox into a zipped, self‑contained HTML
viewer (index.html + images + attachments).

Key features
------------
* **Deduplicate** images/attachments before any processing using SHA‑256.
* **Work‑load preview** – counts unique images first and lets you proceed,
  go back (tweak settings), or quit.
* Optional **GPT‑4o captions** for large images with hardcoded API limit.
* Stores your OpenAI key once in `mbox_to_html_keyDONTSHARE.txt`.
* Creates <o>.zip automatically.

Dependencies
------------
pip install "openai>=1.3.4" requests beautifulsoup4 pillow tqdm

Run
----
python mbox_to_html.py
"""
from __future__ import annotations
import base64
import email
import email.header
import html as html_lib
import json
import mailbox
import mimetypes
import os
import re
import shutil
import sys
import uuid
import hashlib
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Any
from urllib.parse import urlparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
from PIL import Image, UnidentifiedImageError

try:
    from openai import OpenAI            # 1.x client
    OPENAI_V1 = True
except ImportError:
    try:
        import openai                    # 0.x client
        OPENAI_V1 = False
        OpenAI = None
    except ImportError:
        openai = None
        OpenAI = None
        OPENAI_V1 = False

try:
    import tkinter as tk                 # GUI picker
    from tkinter import filedialog
except ImportError:
    tk = None

KEY_FILE = Path(__file__).with_name("mbox_to_html_keyDONTSHARE.txt")

# HARDCODED LIMITS
MAX_CAPTIONS_PER_RUN = 50  # Maximum images to caption with GPT-4o per run

# GPT-4o-mini pricing (as of 2024)
# Input: $0.150 per 1M tokens
# Output: $0.600 per 1M tokens
GPT4O_MINI_INPUT_COST = 0.150 / 1_000_000
GPT4O_MINI_OUTPUT_COST = 0.600 / 1_000_000

# Default caption prompts
SIMPLE_CAPTION_PROMPT = "Describe this image in one concise factual sentence for alt text."

AUTO_DETECT_PROMPT = """Analyze this image thoroughly:
1. First, transcribe ALL visible text exactly as shown (including URLs, names, titles, headings)
2. Then describe the visual elements (people, layout, colors, design)
3. For data/charts: Include all numbers, labels, and values
4. Format as bullet points for clarity

Be comprehensive - never omit text content."""

# Legacy detailed prompt for backwards compatibility
DETAILED_CAPTION_PROMPT = AUTO_DETECT_PROMPT

# Backwards compatibility
DEFAULT_CAPTION_PROMPT = AUTO_DETECT_PROMPT

# Caption length limits
MAX_TOKENS_SIMPLE = 60
MAX_TOKENS_DETAILED = 500  # Much higher for complex content

# ---------- helper utilities ----------
def safe_name(name: str) -> str:
    """Create a safe filename from potentially dangerous input"""
    return re.sub(r"[^A-Za-z0-9_.\\-]", "_", name)[:200] or f"file_{uuid.uuid4().hex}"

def url_ext(url: str) -> str:
    """Extract file extension from URL"""
    ext = os.path.splitext(urlparse(url).path)[1].lower()
    if re.match(r"\.jpe?g(_r)?$", ext): return ".jpeg"
    if re.match(r"\.png(_r)?$", ext):   return ".png"
    return ext if ext in {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"} else ""

def download(url: str, dest: Path, timeout: int = 20) -> bool:
    """Download file from URL with error handling"""
    try:
        r = requests.get(url, timeout=timeout, headers={
            'User-Agent': 'Mozilla/5.0 (compatible; mbox_to_html/4.0)'
        })
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"[WARN] couldn't fetch {url}: {str(e)[:50]}")
        return False

def safe_decode(data: bytes, charset: str = None) -> str:
    """Try multiple encodings to decode email content"""
    if charset:
        try:
            return data.decode(charset)
        except:
            pass
    
    # Try common encodings
    for enc in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1', 'ascii']:
        try:
            return data.decode(enc)
        except:
            continue
    
    return data.decode('utf-8', errors='replace')

def decode_header(header_str: str) -> str:
    """Safely decode email headers"""
    if not header_str:
        return ""
    try:
        decoded_parts = email.header.decode_header(header_str)
        result = []
        for part, encoding in decoded_parts:
            if isinstance(part, bytes):
                result.append(safe_decode(part, encoding))
            else:
                result.append(str(part))
        return ' '.join(result)
    except:
        return str(header_str)

# ---------- OpenAI helpers ----------
def ensure_key() -> Tuple[bool, Optional[Any]]:
    """Ensure OpenAI API key is available"""
    if OPENAI_V1:
        if OpenAI is None:
            print("[ERROR] OpenAI library v1.x not installed. Run: pip install 'openai>=1.3.4'")
            return False, None
    else:
        if openai is None:
            print("[ERROR] OpenAI library not installed. Run: pip install 'openai>=1.3.4'")
            return False, None
    
    key = os.getenv("OPENAI_API_KEY") or (KEY_FILE.read_text().strip() if KEY_FILE.exists() else "")
    if not key:
        key = input("Enter OpenAI API key (blank to skip captions): ").strip()
        if key:
            KEY_FILE.write_text(key)
            os.environ["OPENAI_API_KEY"] = key
    if not key:
        return False, None
    
    try:
        if OPENAI_V1:
            client = OpenAI(api_key=key)
        else:
            # For older openai library
            openai.api_key = key
            client = openai  # Return module itself for 0.x version
        return True, client
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAI client: {e}")
        return False, None

def estimate_image_tokens(img_path: Path) -> int:
    """Estimate tokens for an image based on OpenAI's vision pricing model"""
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            
            # OpenAI uses tiles for image processing
            # Base: 85 tokens, each 512x512 tile: 170 tokens
            # High detail images are first scaled to fit 2048x2048
            
            # Scale down if needed (maintaining aspect ratio)
            max_dim = max(width, height)
            if max_dim > 2048:
                scale = 2048 / max_dim
                width = int(width * scale)
                height = int(height * scale)
            
            # Calculate tiles (512x512 each)
            tiles_x = (width + 511) // 512
            tiles_y = (height + 511) // 512
            total_tiles = tiles_x * tiles_y
            
            # Base cost + tile cost
            tokens = 85 + (170 * total_tiles)
            
            return tokens
    except Exception:
        # Default estimate if we can't open the image
        return 255  # Conservative estimate

def estimate_caption_cost(image_paths: List[Path], prompt: str) -> Tuple[float, int, List[Path]]:
    """Estimate the cost of captioning images after filtering
    Returns: (estimated_cost_usd, number_of_images_to_caption, list_of_images_to_caption)
    """
    # Estimate tokens for the text prompt (rough estimate: 1 token per 4 chars)
    prompt_tokens = len(prompt) // 4 + 10  # +10 for system tokens
    
    total_input_tokens = 0
    images_to_caption = []
    
    for img_path in image_paths[:MAX_CAPTIONS_PER_RUN]:
        if img_path.exists():
            # Input tokens: prompt + image
            total_input_tokens += prompt_tokens + estimate_image_tokens(img_path)
            images_to_caption.append(img_path)
    
    # Output tokens (estimated 60 tokens per caption based on max_tokens)
    total_output_tokens = len(images_to_caption) * 60
    
    # Calculate cost
    input_cost = total_input_tokens * GPT4O_MINI_INPUT_COST
    output_cost = total_output_tokens * GPT4O_MINI_OUTPUT_COST
    total_cost = input_cost + output_cost
    
    return total_cost, len(images_to_caption), images_to_caption

def detect_complex_image(img_path: Path) -> Tuple[bool, str]:
    """Detect if an image is complex (table, chart, text-heavy, etc.)
    Returns: (is_complex, detected_type)
    """
    try:
        with Image.open(img_path) as img:
            width, height = img.size
            
            # Convert to RGB for analysis
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Sample pixels to detect characteristics
            pixels = []
            # Sample in a grid pattern
            sample_points = 100
            for i in range(sample_points):
                x = (i % 10) * (width // 10)
                y = (i // 10) * (height // 10)
                if x < width and y < height:
                    pixels.append(img.getpixel((x, y)))
            
            # Analyze color distribution
            unique_colors = len(set(pixels))
            
            # Get image histogram
            histogram = img.histogram()
            
            # Detect high contrast (common in text/tables)
            # Look for bimodal distribution (text usually has strong peaks at dark and light)
            r_hist = histogram[0:256]
            g_hist = histogram[256:512]
            b_hist = histogram[512:768]
            
            # Simple contrast detection
            total_pixels = width * height
            dark_pixels = sum(r_hist[:50]) / total_pixels
            light_pixels = sum(r_hist[200:]) / total_pixels
            high_contrast = (dark_pixels + light_pixels) > 0.5
            
            # Aspect ratio check (tables often have specific ratios)
            aspect_ratio = width / height if height > 0 else 1
            is_document_like = 0.5 < aspect_ratio < 2.0 and min(width, height) > 400
            
            # File size to dimension ratio (text-heavy images compress differently)
            file_size_kb = img_path.stat().st_size / 1024
            pixels_per_kb = (width * height) / file_size_kb if file_size_kb > 0 else 0
            
            # Detection logic
            reasons = []
            
            if high_contrast and is_document_like:
                reasons.append("table/document")
            
            if unique_colors < 20 and high_contrast:
                reasons.append("chart/diagram")
            
            if pixels_per_kb > 5000 and high_contrast:
                reasons.append("text-heavy")
            
            # Large images with high detail
            if width > 800 and height > 600 and file_size_kb > 200:
                reasons.append("detailed infographic")
            
            is_complex = len(reasons) > 0
            detected_type = " or ".join(reasons) if reasons else "simple image"
            
            return is_complex, detected_type
            
    except Exception as e:
        print(f"[WARN] Failed to analyze image complexity: {e}")
        # Default to simple if we can't analyze
        return False, "unknown"

def need_cap(img: Path, min_kb: int, stats: dict) -> bool:
    """Check if image needs captioning"""
    try:
        size_kb = img.stat().st_size / 1024
        
        with Image.open(img) as im:
            w, h = im.size
            
            # Track why images are skipped
            if w < 96 or h < 96:
                stats['too_small'] += 1
                return False
            
            # Skip potential tracking pixels (1x1, 2x2, etc)
            if w * h < 100:
                stats['tracking_pixel'] += 1
                return False
            
            # Check size requirement
            if size_kb < min_kb:
                stats['under_size_limit'] += 1
                return False
                
            # Check for mostly uniform images (potential spacers/tracking)
            if im.mode in ('RGB', 'RGBA', 'L'):
                # Sample some pixels to check uniformity
                pixels = []
                sample_points = min(100, w * h)  # Limit sampling for performance
                for i in range(sample_points):
                    x = (i * w) // sample_points
                    y = (i * h) // sample_points
                    try:
                        pixels.append(im.getpixel((min(x, w-1), min(y, h-1))))
                    except:
                        pass
                
                # If very few unique colors, skip
                if len(set(pixels)) < 5:
                    stats['uniform_image'] += 1
                    return False
                    
    except (UnidentifiedImageError, OSError) as e:
        stats['unreadable'] += 1
        return False
    except Exception as e:
        print(f"[ERROR] Unexpected error checking image {img}: {e}")
        stats['unreadable'] += 1
        return False
    
    stats['eligible'] += 1
    return True

def caption_img(client: Any, img: Path, caption_count: dict, stats: dict, caption_prompt: str, cost_tracker: dict, use_auto_detect: bool = True) -> str:
    """Caption image with rate limit protection and cost tracking"""
    if caption_count['count'] >= MAX_CAPTIONS_PER_RUN:
        print(f"[LIMIT] Reached max captions ({MAX_CAPTIONS_PER_RUN}). Skipping remaining.")
        stats['skipped_limit'] += 1
        return f"Image {img.name} (caption limit reached)"
    
    caption_count['count'] += 1
    
    # Auto-detect complex images and adjust tokens (but keep same prompt)
    actual_prompt = caption_prompt
    max_tokens = MAX_TOKENS_SIMPLE
    
    if use_auto_detect:
        is_complex, detected_type = detect_complex_image(img)
        if is_complex:
            print(f"   📊 Detected complex image ({detected_type}), using extended token limit...")
            max_tokens = MAX_TOKENS_DETAILED
        else:
            print(f"   📷 Detected simple image, using standard token limit...")
    
    # Estimate cost for this image
    image_tokens = estimate_image_tokens(img)
    prompt_tokens = len(actual_prompt) // 4 + 10
    input_tokens = prompt_tokens + image_tokens
    output_tokens = max_tokens  # Use actual max tokens
    
    image_cost = (input_tokens * GPT4O_MINI_INPUT_COST) + (output_tokens * GPT4O_MINI_OUTPUT_COST)
    cost_tracker['total'] += image_cost
    
    print(f"[AI] Captioning {img.name} ({caption_count['count']}/{MAX_CAPTIONS_PER_RUN}) - Est. cost: ${image_cost:.4f} (Total: ${cost_tracker['total']:.4f})…")
    
    try:
        mime = mimetypes.guess_type(str(img))[0] or "image/jpeg"
        img_data = img.read_bytes()
        
        # Check file size for API limits (OpenAI has a 20MB limit)
        if len(img_data) > 20 * 1024 * 1024:
            print(f"[WARN] Image too large for API: {len(img_data) / 1024 / 1024:.1f}MB")
            return f"Image {img.name} (too large for captioning)"
        
        if OPENAI_V1:
            # New OpenAI API (1.x)
            data_uri = f"data:{mime};base64,{base64.b64encode(img_data).decode()}"
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": actual_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri}}
                    ]
                }],
                max_tokens=max_tokens
            )
            
            # Track actual usage if available
            if hasattr(res, 'usage'):
                actual_input = res.usage.prompt_tokens
                actual_output = res.usage.completion_tokens
                actual_cost = (actual_input * GPT4O_MINI_INPUT_COST) + (actual_output * GPT4O_MINI_OUTPUT_COST)
                print(f"   Actual tokens: {actual_input} in, {actual_output} out (${actual_cost:.4f})")
            
            caption = res.choices[0].message.content.strip()
            
            # For complex images, format the caption nicely
            if max_tokens == MAX_TOKENS_DETAILED and len(caption) > 100:
                # Add some HTML formatting for better readability
                # Convert markdown-style formatting to HTML
                caption = caption.replace("\n\n", "</p><p>")
                caption = caption.replace("\n- ", "</p><ul><li>")
                caption = caption.replace("\n•", "</p><ul><li>")
                if "<ul><li>" in caption:
                    caption = caption.replace("\n", "</li><li>")
                    caption = f"<p>{caption}</li></ul>"
                else:
                    caption = f"<p>{caption}</p>"
                # Clean up any broken tags
                caption = caption.replace("<p></p>", "").replace("<li></li>", "")
                
            return caption
            
        else:
            # Old OpenAI API (0.x) - Note: Vision not supported in 0.x
            print(f"[ERROR] Vision API requires OpenAI library v1.x or newer")
            print(f"[INFO] Please upgrade: pip install 'openai>=1.3.4'")
            return f"Image {img.name} (vision API not available in OpenAI 0.x)"
            
    except Exception as e:
        print(f"[ERROR] Caption failed: {str(e)[:100]}")
        return f"Image {img.name}"

# ---------- SHA‑256 deduper ----------
class Deduper:
    def __init__(self):
        self.hash_map: Dict[str, Path] = {}
        self.duplicate_count = 0
        
    def _hash(self, p: Path) -> str:
        """Calculate SHA-256 hash of file"""
        h = hashlib.sha256()
        try:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 16), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception as e:
            print(f"[ERROR] Failed to hash {p}: {e}")
            return str(uuid.uuid4())  # Return unique value on error
        
    def add(self, p: Path) -> Path:
        """Add file to deduper, return canonical path"""
        h = self._hash(p)
        if h in self.hash_map:
            try:
                p.unlink(missing_ok=True)
                self.duplicate_count += 1
                return self.hash_map[h]
            except Exception as e:
                print(f"[WARN] Failed to remove duplicate {p}: {e}")
                return p
        self.hash_map[h] = p
        return p

# ---------- email helpers ----------
def extract_parts(msg: email.message.Message) -> Tuple[Optional[str], Optional[str], List[email.message.Message]]:
    """Extract HTML body, text body, and attachments from email message"""
    html_body = text_body = None
    attachments: List[email.message.Message] = []
    
    if msg.is_multipart():
        for part in msg.walk():
            disp = part.get_content_disposition()
            ctype = part.get_content_type()
            
            if disp == "attachment" or (disp == "inline" and not ctype.startswith("text/")):
                attachments.append(part)
            elif ctype == "text/html" and html_body is None and disp != "attachment":
                payload = part.get_payload(decode=True)
                if payload:
                    html_body = safe_decode(payload, part.get_content_charset())
            elif ctype == "text/plain" and text_body is None and disp != "attachment":
                payload = part.get_payload(decode=True)
                if payload:
                    text_body = safe_decode(payload, part.get_content_charset())
    else:
        data = msg.get_payload(decode=True)
        if data:
            if msg.get_content_type() == "text/html":
                html_body = safe_decode(data, msg.get_content_charset())
            else:
                text_body = safe_decode(data, msg.get_content_charset())
                
    return html_body, text_body, attachments

# ---------- Phase 1: Download and deduplicate ----------
def phase1_download_and_dedup(mbox_path: Path, out: Path) -> Tuple[Dict[str, Path], Dict[str, str], Deduper]:
    """Phase 1: Download all images and attachments, deduplicate them
    Returns: (image_map, cid_map, deduper)
    """
    print("\n🔽 Phase 1: Downloading and deduplicating files...")
    
    img_dir = out / "images"
    att_dir = out / "attachments"
    img_dir.mkdir(parents=True, exist_ok=True)
    att_dir.mkdir(parents=True, exist_ok=True)
    
    ded = Deduper()
    image_map = {}  # Maps original URL/path to deduplicated file path
    cid_map = {}    # Maps Content-ID to file path
    
    try:
        mbox_file = mailbox.mbox(str(mbox_path))
        total_msgs = len(mbox_file)
        
        for idx, msg in enumerate(tqdm(mbox_file, total=total_msgs, desc="Downloading"), 1):
            try:
                html_body, _, atts = extract_parts(msg)
                
                # Download remote images from HTML
                if html_body:
                    soup = BeautifulSoup(html_body, "html.parser")
                    for im in soup.find_all("img"):
                        src = im.get("src", "")
                        if src.startswith(("http://", "https://")):
                            if src not in image_map:  # Skip if already downloaded
                                fname = safe_name(f"img_{uuid.uuid4().hex}{url_ext(src) or '.img'}")
                                dest = img_dir / fname
                                if download(src, dest):
                                    dest = ded.add(dest)
                                    image_map[src] = dest
                
                # Extract inline/attached images
                for p in atts:
                    data = p.get_payload(decode=True)
                    if not data:
                        continue
                    
                    fname = safe_name(p.get_filename() or f"part_{uuid.uuid4().hex}")
                    dest_dir = img_dir if p.get_content_type().startswith("image/") else att_dir
                    dest = dest_dir / fname
                    
                    try:
                        dest.write_bytes(data)
                        dest = ded.add(dest)
                        
                        # Handle Content-ID for inline images
                        cid_raw = p.get("Content-ID", "").strip("<>").strip()
                        if cid_raw and p.get_content_type().startswith("image/"):
                            cid = cid_raw.split('@')[0] if '@' in cid_raw else cid_raw
                            cid_map[cid] = dest
                            cid_map[cid_raw] = dest  # Also map original
                            
                    except Exception as e:
                        print(f"[ERROR] Failed to save attachment: {e}")
                        
            except Exception as e:
                print(f"[ERROR] Failed to process message {idx}: {e}")
    
    except Exception as e:
        print(f"[ERROR] Failed to open mbox file: {e}")
        raise
    
    print(f"✅ Downloaded {len(image_map)} remote images")
    print(f"✅ Extracted {len([p for p in ded.hash_map.values() if 'images' in str(p)])} unique images")
    print(f"✅ Removed {ded.duplicate_count} duplicates")
    
    return image_map, cid_map, ded

# ---------- Phase 2: Analyze and get caption decisions ----------
def phase2_analyze_and_decide(ded: Deduper, min_kb: int, caption_prompt: str) -> Tuple[bool, List[Path], Optional[Any]]:
    """Phase 2: Analyze images and let user decide on captions
    Returns: (do_captions, images_to_caption, openai_client)
    """
    print("\n📊 Phase 2: Analyzing images...")
    
    # Get all unique images
    all_images = [p for p in ded.hash_map.values() if p.parent.name == "images"]
    
    # Analyze which images are eligible for captions
    caption_stats = {
        'eligible': 0,
        'too_small': 0,
        'tracking_pixel': 0,
        'under_size_limit': 0,
        'uniform_image': 0,
        'unreadable': 0,
        'already_captioned': 0,
        'skipped_limit': 0,
        'download_failed': 0
    }
    
    eligible_images = []
    for img in tqdm(all_images, desc="Analyzing images"):
        if need_cap(img, min_kb, caption_stats):
            eligible_images.append(img)
    
    print(f"\n📊 Image Analysis:")
    print(f"   • Total unique images: {len(all_images)}")
    print(f"   • Eligible for captions: {caption_stats['eligible']}")
    
    if caption_stats['too_small'] > 0:
        print(f"   • Too small (<96px): {caption_stats['too_small']}")
    if caption_stats['tracking_pixel'] > 0:
        print(f"   • Tracking pixels: {caption_stats['tracking_pixel']}")
    if caption_stats['under_size_limit'] > 0:
        print(f"   • Under {min_kb}KB: {caption_stats['under_size_limit']}")
    if caption_stats['uniform_image'] > 0:
        print(f"   • Uniform/blank: {caption_stats['uniform_image']}")
    if caption_stats['unreadable'] > 0:
        print(f"   • Unreadable: {caption_stats['unreadable']}")
    
    if not eligible_images:
        print("\n⚠️  No images eligible for captioning!")
        return False, [], None
    
    # Calculate accurate cost estimate
    images_to_caption = eligible_images[:MAX_CAPTIONS_PER_RUN]
    est_cost, num_images, _ = estimate_caption_cost(images_to_caption, caption_prompt)
    
    print(f"\n💰 Caption Cost Estimate:")
    print(f"   • Images to caption: {num_images}")
    print(f"   • Estimated total cost: ${est_cost:.4f}")
    
    if num_images > 0:
        avg_cost = est_cost / num_images
        print(f"   • Average per image: ${avg_cost:.4f}")
    
    if len(eligible_images) > MAX_CAPTIONS_PER_RUN:
        print(f"\n⚠️  Only first {MAX_CAPTIONS_PER_RUN} images will be captioned (limit)")
    
    # Show some sample images that would be captioned
    print(f"\n📸 Sample images to caption:")
    for i, img in enumerate(images_to_caption[:3]):
        size_kb = img.stat().st_size / 1024
        try:
            with Image.open(img) as im:
                w, h = im.size
                print(f"   {i+1}. {img.name} - {w}x{h}px, {size_kb:.1f}KB")
        except:
            print(f"   {i+1}. {img.name} - {size_kb:.1f}KB")
    
    if len(images_to_caption) > 3:
        print(f"   ... and {len(images_to_caption) - 3} more")
    
    # Ask user to confirm
    confirm = input(f"\n💭 Caption these {num_images} images for ${est_cost:.4f}? [y/N]: ").lower().startswith("y")
    
    if not confirm:
        print("⚠️  Skipping captions")
        return False, [], None
    
    # Get OpenAI client
    ok, client = ensure_key()
    if not ok:
        print("⚠️  Skipping captions (no API key)")
        return False, [], None
    
    return True, images_to_caption, client

# ---------- Phase 3: Caption images ----------
def phase3_caption_images(images_to_caption: List[Path], client: Any, caption_prompt: str, use_auto_detect: bool = True) -> Dict[str, str]:
    """Phase 3: Caption the selected images
    Returns: dict mapping image filename to caption
    """
    print("\n🎨 Phase 3: Generating captions...")
    
    caps = {}
    caption_count = {'count': 0}
    cost_tracker = {'total': 0.0}
    caption_stats = {
        'eligible': len(images_to_caption),
        'too_small': 0,
        'tracking_pixel': 0,
        'under_size_limit': 0,
        'uniform_image': 0,
        'unreadable': 0,
        'already_captioned': 0,
        'skipped_limit': 0,
        'download_failed': 0
    }
    
    for img in tqdm(images_to_caption, desc="Captioning"):
        caption = caption_img(client, img, caption_count, caption_stats, caption_prompt, cost_tracker, use_auto_detect)
        caps[img.name] = caption
    
    print(f"\n✅ Generated {len(caps)} captions")
    print(f"💰 Total cost: ${cost_tracker['total']:.4f}")
    
    return caps

# ---------- Phase 4: Build HTML ----------
def phase4_build_html(
    mbox_path: Path,
    out: Path,
    image_map: Dict[str, Path],
    cid_map: Dict[str, str],
    ded: Deduper,
    caps: Dict[str, str],
    caption_prompt: str
) -> None:
    """Phase 4: Build the final HTML output"""
    print("\n🏗️  Phase 4: Building HTML...")
    
    try:
        mbox_file = mailbox.mbox(str(mbox_path))
        total_msgs = len(mbox_file)
        
        # Generate TOC and messages
        toc_entries = []
        html_parts = []
        
        for idx, msg in enumerate(tqdm(mbox_file, total=total_msgs, desc="Building HTML"), 1):
            try:
                # Generate TOC entry
                subj = decode_header(msg.get("Subject", "(no subject)"))
                date = decode_header(msg.get("Date", ""))
                toc_entries.append(
                    f'<li><a href="#msg{idx}">{html_lib.escape(subj[:80])}{"..." if len(subj) > 80 else ""}</a> '
                    f'<span class="toc-date">({html_lib.escape(date[:20])}...)</span></li>'
                )
                
                # Process message
                html_parts.append(
                    process_message_for_html(idx, msg, out, image_map, cid_map, caps)
                )
            except Exception as e:
                print(f"[ERROR] Failed to process message {idx}: {e}")
                html_parts.append(
                    f'<article id="msg{idx}" class="email-message error">'
                    f'<h2 class="subject">Error processing message {idx}</h2>'
                    f'<div class="body"><p>Failed to process this message: {html_lib.escape(str(e))}</p></div>'
                    f'</article>'
                )
    
    except Exception as e:
        print(f"[ERROR] Failed to open mbox file: {e}")
        raise
    
    # Enhanced CSS with dark mode support
    style = """
        :root {
            --bg-color: #f9f9f9;
            --text-color: #333;
            --card-bg: #fff;
            --border-color: #ddd;
            --link-color: #007bff;
            --meta-color: #666;
            --error-bg: #fee;
            --error-color: #c33;
        }
        
        @media (prefers-color-scheme: dark) {
            :root {
                --bg-color: #1a1a1a;
                --text-color: #e0e0e0;
                --card-bg: #2a2a2a;
                --border-color: #444;
                --link-color: #4dabf7;
                --meta-color: #aaa;
                --error-bg: #4a2c2c;
                --error-color: #ff9999;
            }
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            background: var(--bg-color);
            color: var(--text-color);
            padding: 1rem;
            line-height: 1.6;
        }
        
        .header {
            background: var(--card-bg);
            padding: 2rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,.1);
            margin-bottom: 2rem;
        }
        
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1rem 0;
        }
        
        .stat {
            background: var(--bg-color);
            padding: 1rem;
            border-radius: 4px;
            text-align: center;
        }
        
        .stat-value {
            font-size: 2rem;
            font-weight: bold;
            color: var(--link-color);
        }
        
        .toc {
            background: var(--card-bg);
            padding: 1.5rem;
            border-radius: 8px;
            margin-bottom: 2rem;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .toc h2 {
            margin-top: 0;
        }
        
        .toc ul {
            list-style: none;
            padding: 0;
        }
        
        .toc li {
            padding: 0.5rem 0;
            border-bottom: 1px solid var(--border-color);
        }
        
        .toc-date {
            color: var(--meta-color);
            font-size: 0.85em;
        }
        
        .email-message {
            background: var(--card-bg);
            padding: 1.5rem;
            margin: 1rem 0;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0,0,0,.1);
        }
        
        .email-message.error {
            border: 2px solid var(--error-color);
        }
        
        .email-message h2 {
            margin-top: 0;
            color: var(--text-color);
        }
        
        .meta {
            color: var(--meta-color);
            font-size: 0.9em;
            margin: 1rem 0;
            padding: 0.5rem;
            background: var(--bg-color);
            border-radius: 4px;
        }
        
        .attachments {
            background: var(--bg-color);
            padding: 0.75rem;
            border-radius: 4px;
            margin: 1rem 0;
        }
        
        .body {
            margin-top: 1rem;
        }
        
        .body img {
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }
        
        figure {
            margin: 1rem 0;
            text-align: center;
        }
        
        figcaption {
            font-size: 0.9em;
            color: var(--meta-color);
            font-style: italic;
            margin-top: 0.5rem;
            text-align: left;
            max-width: 100%;
        }
        
        /* Style for detailed captions */
        figcaption p {
            margin: 0.5rem 0;
            font-style: normal;
        }
        
        figcaption ul {
            margin: 0.5rem 0;
            padding-left: 1.5rem;
            text-align: left;
            font-style: normal;
        }
        
        figcaption li {
            margin: 0.25rem 0;
        }
        
        /* For very long captions, add some styling */
        figcaption.detailed {
            background: var(--bg-color);
            padding: 1rem;
            border-radius: 4px;
            margin-top: 0.5rem;
            font-size: 0.85em;
        }
        
        .missing-image {
            background: var(--error-bg);
            color: var(--error-color);
            padding: 0.5rem;
            border-radius: 4px;
            font-size: 0.85em;
        }
        
        pre {
            overflow-x: auto;
            background: var(--bg-color);
            padding: 1rem;
            border-radius: 4px;
        }
        
        a {
            color: var(--link-color);
            text-decoration: none;
        }
        
        a:hover {
            text-decoration: underline;
        }
        
        hr {
            border: none;
            border-top: 1px solid var(--border-color);
            margin: 2rem 0;
        }
    """
    
    # Build final HTML
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{mbox_path.stem} - Email Archive</title>
    <style>{style}</style>
</head>
<body>
    <div class="header">
        <h1>{mbox_path.stem} Email Archive</h1>
        <div class="stats">
            <div class="stat">
                <div class="stat-value">{len(html_parts)}</div>
                <div>Total Messages</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(ded.hash_map)}</div>
                <div>Unique Files</div>
            </div>
            <div class="stat">
                <div class="stat-value">{ded.duplicate_count}</div>
                <div>Duplicates Removed</div>
            </div>
            <div class="stat">
                <div class="stat-value">{len(caps)}</div>
                <div>Images Captioned</div>
            </div>
        </div>
        <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            {''.join(toc_entries)}
        </ul>
    </div>
    
    <div class="messages">
        {'<hr>'.join(html_parts)}
    </div>
</body>
</html>"""
    
    out.joinpath("index.html").write_text(html, "utf-8")

    if caps:
        caps_data = {
            "captions": caps,
            "prompt": caption_prompt,
            "generated_at": datetime.now().isoformat()
        }
        out.joinpath("captions.json").write_text(json.dumps(caps_data, indent=2), "utf-8")

def process_message_for_html(
    idx: int,
    msg: email.message.Message,
    out: Path,
    image_map: Dict[str, Path],
    cid_map: Dict[str, str],
    caps: Dict[str, str]
) -> str:
    """Process single email message for HTML output"""
    subj = decode_header(msg.get("Subject", "(no subject)"))
    frm  = decode_header(msg.get("From", ""))
    to   = decode_header(msg.get("To", ""))
    date = decode_header(msg.get("Date", ""))

    html_body, text_body, atts = extract_parts(msg)

    if html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        
        # Replace image sources with local paths
        for im in soup.find_all("img"):
            src = im.get("src", "")
            
            if src.startswith("cid:"):
                # Handle CID references
                cid = src[4:]  # Remove "cid:" prefix
                if cid in cid_map:
                    im["src"] = str(cid_map[cid].relative_to(out))
                    # Add caption if available
                    if cid_map[cid].name in caps:
                        alt = caps[cid_map[cid].name]
                        im["alt"] = alt if len(alt) < 1000 else alt[:1000] + "..."  # Truncate alt text at higher limit
                        # Create figure with caption
                        fig = soup.new_tag("figure")
                        im.replace_with(fig)
                        fig.append(im)
                        cap_tag = soup.new_tag("figcaption")
                        
                        # Check if it's a detailed caption
                        if len(alt) > 200 or "<p>" in alt or "<ul>" in alt:
                            cap_tag["class"] = "detailed"
                            # Insert HTML directly for formatted captions
                            cap_tag.append(BeautifulSoup(alt, "html.parser"))
                        else:
                            cap_tag.string = alt
                        
                        fig.append(cap_tag)
            
            elif src in image_map:
                # Handle remote images
                dest = image_map[src]
                im["src"] = str(dest.relative_to(out))
                # Add caption if available
                if dest.name in caps:
                    alt = caps[dest.name]
                    im["alt"] = alt if len(alt) < 200 else alt[:200] + "..."  # Truncate alt text
                    # Create figure with caption
                    fig = soup.new_tag("figure")
                    im.replace_with(fig)
                    fig.append(im)
                    cap_tag = soup.new_tag("figcaption")
                    
                    # Check if it's a detailed caption
                    if len(alt) > 200 or "<p>" in alt or "<ul>" in alt:
                        cap_tag["class"] = "detailed"
                        # Insert HTML directly for formatted captions
                        cap_tag.append(BeautifulSoup(alt, "html.parser"))
                    else:
                        cap_tag.string = alt
                    
                    fig.append(cap_tag)
            
            elif src.startswith(("http://", "https://")):
                # Image wasn't downloaded successfully
                ph = soup.new_tag("div", attrs={"class": "missing-image"})
                ph.string = f"[Image not accessible: {src[:50]}...]"
                im.replace_with(ph)
                
        body_html = str(soup)
    else:
        body_html = f"<pre>{html_lib.escape(text_body or '(no content)')}</pre>"

    # Handle non-image attachments
    att_links = []
    att_dir = out / "attachments"
    
    for p in atts:
        if not p.get_content_type().startswith("image/"):
            fname = safe_name(p.get_filename() or f"part_{uuid.uuid4().hex}")
            # Check if this attachment exists
            att_path = att_dir / fname
            if att_path.exists():
                att_links.append(
                    f'<a href="{att_path.relative_to(out)}">{html_lib.escape(fname)}</a>'
                )

    att_block = (
        f'<p class="attachments"><strong>Attachments:</strong> {", ".join(att_links)}</p>'
        if att_links else ""
    )

    return (
        f'<article id="msg{idx}" class="email-message">'
        f'<h2 class="subject">{html_lib.escape(subj)}</h2>'
        f'<div class="meta">'
        f'<span class="date">{html_lib.escape(date)}</span><br>'
        f'<span class="from">From: {html_lib.escape(frm)}</span><br>'
        f'<span class="to">To: {html_lib.escape(to)}</span>'
        f'</div>'
        f'{att_block}'
        f'<div class="body">{body_html}</div>'
        f'</article>'
    )

# ---------- workload estimator ----------
def estimate(mbox_path: Path) -> Tuple[int, int, int]:
    """Returns (remote_images, inline_images, approx_unique_after_dedup)"""
    remote = inline = 0
    size_hash_map = {}  # Quick dedup estimate based on size
    
    print("Analyzing mbox file...")
    try:
        mbox_file = mailbox.mbox(str(mbox_path))
        total = len(mbox_file)
        
        for msg in tqdm(mbox_file, total=total, desc="Analyzing"):
            try:
                html_body, _, atts = extract_parts(msg)
                if html_body:
                    soup = BeautifulSoup(html_body, "html.parser")
                    remote += sum(
                        1 for im in soup.find_all("img")
                        if im.get("src", "").startswith(("http://", "https://"))
                    )
                
                for p in atts:
                    if p.get_content_type().startswith("image/"):
                        inline += 1
                        # Rough dedup estimate based on size
                        payload = p.get_payload(decode=True)
                        if payload:
                            size = len(payload)
                            size_hash_map[size] = size_hash_map.get(size, 0) + 1
            except Exception as e:
                print(f"[WARN] Failed to analyze message: {e}")
                continue
        
        # Estimate duplicates (same size likely = same image)
        estimated_dups = sum(count - 1 for count in size_hash_map.values() if count > 1)
        approx_unique = max(1, inline - estimated_dups)
        
        return remote, inline, approx_unique
    except Exception as e:
        print(f"[ERROR] Failed to open mbox file: {e}")
        return 0, 0, 0

# ---------- file picker & wizard ----------
def pick_file() -> Optional[Path]:
    """Pick file using GUI dialog or command line input"""
    if tk:
        try:
            root = tk.Tk()
            root.withdraw()
            p = filedialog.askopenfilename(
                title="Select .mbox file",
                filetypes=[("Mailbox files","*.mbox"),("All files","*.*")],
            )
            root.destroy()
            if p:
                return Path(p)
        except Exception as e:
            print(f"[WARN] GUI file picker failed: {e}")
    
    path_str = input("Path to .mbox file: ").strip()
    if path_str:
        return Path(path_str)
    return None

def get_unique_output_path(base_name: str) -> Path:
    """Get a unique output path by appending (1), (2), etc. if needed"""
    base_path = Path(base_name)
    if not base_path.exists() and not Path(f"{base_path}.zip").exists():
        return base_path
    
    counter = 1
    while True:
        new_path = Path(f"{base_name}({counter})")
        if not new_path.exists() and not Path(f"{new_path}.zip").exists():
            return new_path
        counter += 1

def wizard():
    """Interactive wizard for converting mbox to HTML"""
    print("=" * 60)
    print("MBOX to HTML Converter v4.0")
    print("=" * 60)
    
    mbox = pick_file()
    if not mbox or not mbox.exists():
        sys.exit("❌ File not found")

    print(f"\n📧 Selected: {mbox.name}")
    print(f"📊 File size: {mbox.stat().st_size / 1024 / 1024:.1f} MB")

    # Initial configuration
    print("\n" + "-" * 40)
    
    caption_prompt = AUTO_DETECT_PROMPT
    use_auto_detect = True
    cap_enabled = input("Enable image captions with GPT-4o-mini? [y/N]: ").lower().startswith("y")
    
    if cap_enabled:
        print(f"\n💡 Caption modes:")
        print(f"   1. Auto-detect mode (Recommended): Intelligently adapts to image type")
        print(f"   2. Simple mode: Brief one-sentence descriptions only")
        print(f"   3. Custom mode: Use your own prompt")
        
        mode = input("\nSelect mode [1/2/3] (Enter for auto-detect): ").strip() or "1"
        
        if mode == "2":
            caption_prompt = SIMPLE_CAPTION_PROMPT
            use_auto_detect = False
            print(f"📝 Using simple captions")
        elif mode == "3":
            print(f"\n💡 Enter your custom prompt:")
            caption_prompt = input("> ").strip() or SIMPLE_CAPTION_PROMPT
            use_auto_detect = False
            print(f"📝 Using custom prompt")
        else:  # mode == "1" or default
            use_auto_detect = True
            print(f"\n📝 Using auto-detect mode (recommended)")
            print(f"💡 Default prompt: \"{AUTO_DETECT_PROMPT[:80]}...\"")
            
            # Allow customization of the auto-detect prompt
            customize = input("\nCustomize this prompt? [y/N]: ").lower().startswith("y")
            if customize:
                print("\nEnter your custom auto-detect prompt:")
                caption_prompt = input("> ").strip() or AUTO_DETECT_PROMPT
            else:
                caption_prompt = AUTO_DETECT_PROMPT
            
            print(f"\n✅ Auto-detect will use:")
            print(f"   • Simple images: {MAX_TOKENS_SIMPLE} tokens max")
            print(f"   • Complex images: {MAX_TOKENS_DETAILED} tokens max")
    
    kb = input("\nMinimum image file size in KB to process (Enter for 10): ").strip()
    min_kb = int(kb) if kb.isdigit() else 10

    # Quick estimate (before downloading)
    r, i, u = estimate(mbox)
    print(f"\n📊 Quick Analysis:")
    print(f"   • Remote images to download: ~{r}")
    print(f"   • Inline/attached images: ~{i}")
    print(f"   • Estimated unique images: ~{u}")
    print(f"   • Total images to process: ~{r + u}")
    
    if cap_enabled:
        print(f"\n💡 Note: We'll download all images first, then show exact")
        print(f"   caption costs based on actual file sizes and dimensions.")

    ch = input("\n✅ Proceed? [Y/n]: ").lower()
    if ch == "n":
        sys.exit("Cancelled by user")

    # Get output path
    suggested_name = input("\nOutput folder name (Enter for 'mbox_viewer'): ").strip() or "mbox_viewer"
    out = get_unique_output_path(suggested_name)
    
    if out.name != suggested_name:
        print(f"📁 Using '{out.name}' ('{suggested_name}' already exists)")

    print(f"\n🚀 Starting conversion...")
    start_time = datetime.now()
    
    try:
        # Phase 1: Download and deduplicate
        image_map, cid_map, ded = phase1_download_and_dedup(mbox, out)
        
        # Phase 2: Analyze and get caption decisions (only if captions enabled)
        caps = {}
        if cap_enabled:
            do_captions, images_to_caption, client = phase2_analyze_and_decide(ded, min_kb, caption_prompt)
            
            # Phase 3: Caption images (if approved)
            if do_captions and client and images_to_caption:
                caps = phase3_caption_images(images_to_caption, client, caption_prompt, use_auto_detect)
        
        # Phase 4: Build HTML
        phase4_build_html(mbox, out, image_map, cid_map, ded, caps, caption_prompt)
        
        # Create zip file
        print(f"\n📦 Creating {out}.zip...")
        shutil.make_archive(str(out), "zip", out)
        
        duration = (datetime.now() - start_time).total_seconds()
        print(f"\n✅ Success! Completed in {duration:.1f} seconds")
        print(f"📦 Output: {out}.zip")
        print(f"🌐 Open {out}/index.html in your browser")
        
    except KeyboardInterrupt:
        print("\n\n❌ Cancelled by user")
        # Clean up partial output
        if out.exists():
            shutil.rmtree(out)
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Clean up partial output
        if out.exists():
            shutil.rmtree(out)
        sys.exit(1)

if __name__ == "__main__":
    wizard()
