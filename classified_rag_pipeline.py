import requests
from bs4 import BeautifulSoup
from google import genai
from google.genai import types
import os
import sys
import time
import re
import json
from collections import defaultdict
from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
from supabase import create_client, Client

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Configuration ---
CATEGORIES = [
    # --- Application Layer ---
    'Agents',          # Autonomous systems, tool use, multi-agent flows
    'RAG',             # Retrieval Augmented Generation patterns
    'Prompting',       # Prompt engineering and optimization
    'Salesforce-agentforce',
    
    # --- Model Layer ---
    'LLM-Model',       # Base models (Llama 3, GPT-4, Mistral)
    'Multimodal',      # Image/Video/Audio generation and analysis
    'Fine-Tuning',     # Training, LoRA, RLHF, DPO
    
    # --- Infrastructure & Tools ---
    'Framework',       # Orchestration (LangChain, LlamaIndex)
    'Vector-DB',       # Embedding storage (Chroma, Pinecone)
    'LLM-Eval',        # Benchmarks and evaluation metrics
    'Inference',       # Serving engines, quantization, local hosting
    'Ops',             # Monitoring, deployment, logging
    
    # --- Other ---
    'Hardware',        # GPUs, TPUs, local compute
    'Other'            # Catch-all
    'Global'           # All by default.
]

BATCH_SIZE = 10

CONTENT_DIR = "scraped_content"
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

# --- 1. Web Scraping Functions ---
def get_youtube_video_id(url):
    """Extracts video ID from YouTube URL."""
    parsed = urlparse(url)
    if parsed.hostname == 'youtu.be':
        return parsed.path[1:]
    if parsed.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed.path == '/watch':
            p = parse_qs(parsed.query)
            return p['v'][0] if 'v' in p else None
        if parsed.path.startswith('/embed/'):
            return parsed.path.split('/')[2]
        if parsed.path.startswith('/v/'):
            return parsed.path.split('/')[2]
    return None

def scrape_youtube(url):
    """Scrapes transcript and metadata from a YouTube video."""
    video_id = get_youtube_video_id(url)
    if not video_id:
        return None, None
        
    print(f"  Detected YouTube Video ID: {video_id}")
    
    text_content = ""
    title = f"YouTube Video {video_id}" # Default title
    
    # 1. Get Transcript
    try:
        # In this environment, YouTubeTranscriptApi is a class that needs instantiation
        api = YouTubeTranscriptApi()
        transcript = api.fetch(video_id)       
        # Handle transcript being a list of dicts or an object with snippets
        iterator = transcript.snippets if hasattr(transcript, 'snippets') else transcript
        text_content = " ".join([entry['text'] if isinstance(entry, dict) else entry.text for entry in iterator])
        print(f"  Fetched transcript ({len(text_content)} chars).")
    except Exception as e:
        print(f"  Could not fetch transcript for {url}: {e}", file=sys.stderr)
        return None, None # If no transcript, we probably don't want to index just the title? Or maybe we do? 
                          # User asked: "scrape the transcript if available else just the title , description"
                          # So if transcript fails, we continue to try metadata.
    
    # 2. Get Metadata (Title/Description) via simple scraping
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            page_title = soup.find('title')
            if page_title:
                title = page_title.get_text(strip=True).replace(" - YouTube", "")
            
            # Description is harder to get reliably from raw HTML without JS, 
            # but sometimes it's in meta tags.
            meta_desc = soup.find('meta', attrs={'name': 'description'})
            if meta_desc:
                description = meta_desc.get('content')
                text_content = f"Description: {description}\n\nTranscript:\n{text_content}"
            else:
                text_content = f"Transcript:\n{text_content}"
                
    except Exception as e:
        print(f"  Could not fetch metadata for {url}: {e}", file=sys.stderr)

    return title, text_content

def scrape_url(url):
    """Scrapes text content from a URL, handling YouTube specially."""
    
    # Check for YouTube
    if "youtube.com" in url or "youtu.be" in url:
        return scrape_youtube(url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to find the main content
        main_content = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main_content:
            title = soup.find('title').get_text(strip=True) if soup.find('title') else "No Title"
            # Remove scripts and styles
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            text = main_content.get_text(separator=' ', strip=True)
            return title, text
        else:
            return "No Title", ""
            
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}", file=sys.stderr)
        return None, None

def safe_filename(title):
    """Creates a safe filename from a title."""
    title = re.sub(r'[^\w\s-]', '', title).strip().lower()
    title = re.sub(r'[-\s]+', '-', title)
    return title[:50]

def batch_process(iterable, n=1):
    """Yields successive n-sized chunks from iterable."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

# --- 2. Classification Function ---
def classify_content_batch(client, file_contents):
    """
    Classifies a batch of file contents using Gemini.
    file_contents: dict {filename: content_text}
    Returns: dict {filename: [category1, category2, ...]}
    """
    if not file_contents:
        return {}

    prompt = f"""
    You are a document classifier. 
    Classify the following documents into one or more of these categories: {CATEGORIES}.
    
    If a document fits multiple categories, list all of them.
    If a document does not fit any of the specific categories, use 'Other'.
    
    Input Documents (Format: Filename: Content Snippet):
    """
    
    for filename, content in file_contents.items():
        # Truncate content to avoid context limit issues in classification (first 1000 chars is usually enough)
        snippet = content[:1000].replace('\n', ' ')
        prompt += f"\n--- DOCUMENT START: {filename} ---\n{snippet}\n--- DOCUMENT END ---\n"
        
    prompt += f"""
    \nReturn ONLY a raw JSON object (no markdown formatting) mapping each filename to a list of categories.
    Example format:
    {{
        "file1.txt": ["RAG", "Agents"],
        "file2.txt": ["Other"]
    }}
    """

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-exp", # Using a fast model for classification
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )
        
        # Clean up response text just in case
        text = response.text.strip()
        if text.startswith("```json"):
            text = text[7:-3]
        
        return json.loads(text)
    except Exception as e:
        print(f"Error classifying batch: {e}", file=sys.stderr)
        # Fallback: assign 'Other' to all in this batch
        return {fname: ['Other'] for fname in file_contents.keys()}

# --- Main Pipeline ---
def main():
    print("Starting Classified RAG Pipeline...")
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: API Key not found.")
        return
        
    client = genai.Client(api_key=api_key)
    
    # Initialize Supabase
    print("Initializing Supabase...")
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    # Read existing config for category-store map
    category_store_map = {}
    current_config = {}
    try:
        response = supabase.table('agent_config').select('config').eq('agent_type', 'control_agent').execute()
        if response.data:
            config_data = response.data[0].get('config')
            if config_data:
                current_config = config_data if isinstance(config_data, dict) else json.loads(config_data)
                category_store_map = current_config.get('category_store_map', {})
                print(f"Loaded existing category-store map: {category_store_map}")
    except Exception as e:
        print(f"Error reading agent_config: {e}")

    print("\n Get existing file stores...")
    existing_stores = client.file_search_stores.list()
    existing_store_names = {store.display_name for store in existing_stores}
    print(existing_store_names)
    
    #sys.exit()


    # --- Step 1 & 2: Batch Scrape ---
    # --- Step 1 & 2: Batch Scrape ---
    print("Fetching links from Supabase...")
    urls = []
    current_row_id = None
    try:
        # Fetch latest row with content=null
        response = supabase.table('scraper_data') \
            .select('id, original_links') \
            .is_('content', 'null') \
            .order('created_at', desc=True) \
            .limit(1) \
            .execute()
            
        if response.data:
            row = response.data[0]
            current_row_id = row.get('id')
            original_links = row.get('original_links')
            if isinstance(original_links, list):
                urls = original_links
            elif isinstance(original_links, str):
                try:
                    urls = json.loads(original_links)
                except:
                    print("Error parsing original_links JSON")
                    pass
            
            if not urls:
                print("Found row but no links in original_links.")
                return

            print(f"Fetched {len(urls)} links from Supabase (Row ID: {current_row_id}).")
        else:
            print("No pending jobs (content=null) found in Supabase.")
            return

    except Exception as e:
        print(f"Error fetching links from Supabase: {e}")
        return

    os.makedirs(CONTENT_DIR, exist_ok=True)
    
    print(f"Found {len(urls)} links. Starting batch scraping...")
    
    # We need to keep track of generated files for the next step
    generated_files = [] # List of full paths

    for batch_idx, url_batch in enumerate(batch_process(urls, BATCH_SIZE)):
        print(f"Processing Scraping Batch {batch_idx + 1}...")
        for url in url_batch:
            title, content = scrape_url(url)
            if content:
                filename = safe_filename(title) or f"doc_{int(time.time())}"
                filepath = os.path.join(CONTENT_DIR, f"{filename}.txt")
                
                # Avoid overwriting if title is same/generic, append timestamp if needed
                if os.path.exists(filepath):
                     filepath = os.path.join(CONTENT_DIR, f"{filename}_{int(time.time())}.txt")

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"Source URL: {url}\n")
                    f.write(f"Title: {title}\n\n")
                    f.write(content)
                
                generated_files.append(filepath)
                print(f"  Saved: {os.path.basename(filepath)}")
            else:
                print(f"  Skipped (no content): {url}")

    if not generated_files:
        print("No content scraped. Exiting.")
        return

    # --- Step 3: Batch Classification ---
    print("\nStarting Batch Classification...")
    
    # Map filename -> [categories]
    file_categories = {}
    
    # Read files and classify in batches
    for batch_idx, file_batch in enumerate(batch_process(generated_files, BATCH_SIZE)):
        print(f"Processing Classification Batch {batch_idx + 1}...")
        
        batch_contents = {}
        for filepath in file_batch:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    batch_contents[os.path.basename(filepath)] = f.read()
            except Exception as e:
                print(f"  Error reading {filepath}: {e}")
        
        if batch_contents:
            results = classify_content_batch(client, batch_contents)
            file_categories.update(results)
            print(f"  Classified {len(results)} files.")

    # --- Step 4 & 5: Upload and Vectorize ---
    print("\nStarting Upload and Vectorization...")
    
    # 1. Upload ALL files to Gemini (we need the file resource names)
    # Map local_filename -> gemini_file_object
    uploaded_files_map = {} 
    
    print("Uploading files to Gemini...")
    for filepath in generated_files:
        filename = os.path.basename(filepath)
        try:
            print(f"  Uploading {filename}...")
            gemini_file = client.files.upload(file=filepath, config=types.UploadFileConfig(display_name=filename))
            uploaded_files_map[filename] = gemini_file
        except Exception as e:
            print(f"  Failed to upload {filename}: {e}")

    # 2. Group files by category
    # Category -> List of gemini_file_objects
    category_groups = defaultdict(list)
    
    for filename, categories in file_categories.items():
        if filename in uploaded_files_map:
            gemini_file = uploaded_files_map[filename]
            for cat in categories:
                # Normalize category name just in case
                if cat in CATEGORIES:
                    category_groups[cat].append(gemini_file)                    
                else:
                    category_groups['Other'].append(gemini_file)

            # Add Global category
            category_groups['Global'].append(gemini_file)
        else:
            print(f"Warning: {filename} was classified but upload failed/missing.")


    
    # 3. Create Vector Store per Category
    print("\nCreating Vector Stores...")
    
    for category, files in category_groups.items():
        if not files:
            continue
            
        print(f"  Processing Category: {category} ({len(files)} files)")
        
        try:
            # Check if store already exists in map
            store_name = category_store_map.get(category)
            
            if store_name:
                print(f"    Using existing store for {category}: {store_name}")
            else:
                # Create store
                display_name = f"VS_{category}" 
                store = client.file_search_stores.create(
                    config=types.CreateFileSearchStoreConfig(display_name=display_name)
                )
                store_name = store.name
                category_store_map[category] = store_name
                print(f"    Created Store: {store.name} ({display_name})")
            
            # Import files
            for file in files:
                client.file_search_stores.import_file(
                    file_search_store_name=store_name, 
                    file_name=file.name
                )
            print(f"    Added {len(files)} files to {category} store.")
            
        except Exception as e:
            print(f"    Error processing store for {category}: {e}")

    # Update Supabase with new map
    print(f"\nPreparing to update Supabase. Map to save: {category_store_map}")
    try:
        current_config['category_store_map'] = category_store_map
        response = supabase.table('agent_config').update({'config': current_config}).eq('agent_type', 'control_agent').execute()
        print(f"Supabase Update Response: {response}")
        if response.data:
            print("Updated agent_config with new category-store map.")
        else:
            print("Warning: No rows were updated. Check if 'agent_type=control_agent' exists.")
    except Exception as e:
        print(f"Error updating agent_config: {e}")

    # --- Step 6: Save to Supabase ---
    print("\nSaving to Supabase...")
    try:
        # supabase client already initialized

        
        scraper_data_list = []
        
        for filepath in generated_files:
            filename = os.path.basename(filepath)
            
            # Read file to extract URL and Content
            # Format in file:
            # Source URL: <url>
            # Title: <title>
            # <content>
            
            file_url = ""
            file_content = ""
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    
                # Extract URL (assuming first line)
                if lines and lines[0].startswith("Source URL:"):
                    file_url = lines[0].replace("Source URL:", "").strip()
                
                # Extract Content (skip headers)
                # Find where content starts (after Title and empty line)
                content_start_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith("Title:"):
                        content_start_idx = i + 2 # Skip Title and next newline
                        break
                
                if content_start_idx < len(lines):
                    file_content = "".join(lines[content_start_idx:])
                else:
                    file_content = "".join(lines) # Fallback
                    
            except Exception as e:
                print(f"  Error reading {filename} for Supabase: {e}")
                continue

            # Get Categories
            cats = file_categories.get(filename, ['Other'])
            cats_str = ", ".join(cats)
            
            scraper_data_list.append({
                "link": file_url,
                "categories": cats_str,
                "content": file_content
            })
            
        if scraper_data_list:
            if current_row_id:
                print(f"  Updating Row ID {current_row_id} with content...")
                data, count = supabase.table('scraper_data').update({"content": scraper_data_list}).eq('id', current_row_id).execute()
                print(f"  Successfully updated row {current_row_id} in Supabase.")
            else:
                print("  Error: current_row_id is missing. Cannot update Supabase.")
        else:
            print("  No data to save to Supabase.")
            
    except Exception as e:
        print(f"  Error saving to Supabase: {e}")

    print("\nPipeline Complete!")

if __name__ == "__main__":
    main()
