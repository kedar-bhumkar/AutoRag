# AutoRag: Automated Classified RAG Pipeline

AutoRag is an automated pipeline that scrapes content from URLs (including YouTube transcripts), classifies the content using an LLM (Google Gemini), and uploads it to curated Vector Stores for RAG (Retrieval Augmented Generation) applications. It manages state via Supabase and is designed to run both locally and via GitHub Actions.

## Features

*   **Smart Scraping**: Extracts text from standard webpages and fetches transcripts from YouTube videos automatically.
*   **Intelligent Classification**: Uses Google Gemini (`gemini-2.0-flash-exp`) to classify documents into predefined categories (e.g., Agents, RAG, Prompting, Vector-DB).
*   **Vector Store Management**: Automatically creates and manages Gemini File Search Vector Stores based on content categories.
*   **State Management**: Reads pending links from and writes processed data back to a Supabase database.
*   **Automated Workflow**: Includes a GitHub Actions workflow (`scraper.yml`) to run the pipeline on a schedule.

## Prerequisites

*   Python 3.11+
*   A Supabase project
*   Google Gemini API Key

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd AutoRag
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Setup:**
    Create a `.env` file in the root directory:
    ```ini
    SUPABASE_URL=your_supabase_url
    SUPABASE_KEY=your_supabase_anon_key
    GEMINI_API_KEY=your_google_ai_studio_key
    ```

## Usage

### Running Locally
To run the pipeline manually:
```bash
python classified_rag_pipeline.py
```

### Running on GitHub Actions
The project includes a workflow at `.github/workflows/scraper.yml` that runs:
*   **Schedule**: Every hour.
*   **Manual**: Can be triggered via the "Run workflow" button in the Actions tab.

**Important**: You must add the following secrets to your GitHub Repository (Settings -> Secrets and variables -> Actions):
*   `SUPABASE_URL`
*   `SUPABASE_KEY`
*   `GEMINI_API_KEY`

## Configuration

### Database Schema (Supabase)

The pipeline expects two tables in your Supabase project:

1.  **`scraper_data`**: Used to fetch links and store results.
    *   `id`: unique identifier.
    *   `original_links`: JSON array or list of URLs to process.
    *   `content`: JSON field where the scraped and classified result is stored.
    *   `created_at`: timestamp.

2.  **`agent_config`**: Stores the mapping between categories and Vector Store IDs.
    *   `agent_type`: text (should contain a row where value is `'control_agent'`).
    *   `config`: JSON field containing `category_store_map`.

### Categories
Categories are defined in `classified_rag_pipeline.py`. To modify them, edit the `CATEGORIES` list in the script.

## Directory Structure

*   `classified_rag_pipeline.py`: Main logic script.
*   `scraped_content/`: Temporary directory for downloaded text files.
*   `.github/workflows/scraper.yml`: GitHub Actions definition.
*   `requirements.txt`: Python dependencies.
