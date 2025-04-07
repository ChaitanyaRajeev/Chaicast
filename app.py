"""
Chaicast Flask Web Application - Turn articles into podcasts
"""

import os
import uuid
import json
import time
import logging
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_file, abort
from functools import wraps
from dotenv import load_dotenv
from podcast import convert_text_to_podcast, create_output_directories
import threading

# Load environment variables from .env file
load_dotenv()

# Create directories for jobs, transcripts, and audio files
BASE_DIR = Path(__file__).resolve().parent
JOBS_DIR = BASE_DIR / "jobs"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
AUDIO_DIR = BASE_DIR / "audio"

for directory in [JOBS_DIR, TRANSCRIPT_DIR, AUDIO_DIR]:
    directory.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chaicast_flask")

# Initialize Flask app
app = Flask(__name__)
app.config['DEBUG'] = True

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "default-insecure-key")  # Default for development only

# API key authorization decorator
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Check if API key is provided in header or as a query parameter
        provided_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        # For form submissions, also check form data
        if request.method == 'POST' and not provided_key:
            provided_key = request.form.get('api_key')
            
        # If no key provided or key doesn't match, return 401 Unauthorized
        if not provided_key or provided_key != API_KEY:
            return jsonify({"error": "Unauthorized. Invalid or missing API key."}), 401
            
        return f(*args, **kwargs)
    return decorated_function

# Enable CORS for browser preview
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Add template filter for timestamp conversion
@app.template_filter('timestamp_to_datetime')
def timestamp_to_datetime(timestamp):
    """Convert Unix timestamp to readable datetime"""
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

# In-memory job storage (in a real app, you'd use a database)
JOBS = {}

def load_jobs():
    """Load existing jobs from the jobs directory"""
    try:
        if not JOBS_DIR.exists():
            return
            
        for job_file in JOBS_DIR.glob("*.json"):
            try:
                with open(job_file, "r") as f:
                    job_data = json.load(f)
                    JOBS[job_data["id"]] = job_data
                    logger.info(f"Loaded job {job_data['id']} with status {job_data['status']}")
            except Exception as e:
                logger.error(f"Error loading job from {job_file}: {str(e)}")
    except Exception as e:
        logger.error(f"Error loading jobs: {str(e)}")

def get_job_data(job_id):
    """Get job data from memory or disk"""
    if job_id in JOBS:
        return JOBS[job_id]
    else:
        job_file = JOBS_DIR / f"{job_id}.json"
        if job_file.exists():
            with open(job_file, "r") as f:
                job_data = json.load(f)
                JOBS[job_id] = job_data  # Update in-memory cache
                return job_data
        else:
            return None

def process_podcast_job(job_id, text, tts_model="openai", custom_intro=""):
    """Process a podcast generation job"""
    try:
        # Get API keys from environment variables
        openai_api_key = os.getenv("OPENAI_API_KEY")
        elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        
        # Call the podcast generation function
        transcript_path, audio_path = convert_text_to_podcast(
            text=text,
            tts_model=tts_model,
            custom_intro=custom_intro,
            openai_api_key=openai_api_key,
            elevenlabs_api_key=elevenlabs_api_key
        )
        
        # Update job status to completed
        result = {
            "transcript": str(transcript_path),
            "audio": str(audio_path)
        }
        update_job_status(job_id, "completed", result=result)
        
        return True
    except Exception as e:
        logger.error(f"Error processing job {job_id}: {str(e)}")
        update_job_status(job_id, "failed", error=str(e))
        return False

def update_job_status(job_id, status, result=None, error=None):
    """Update the status of a job"""
    try:
        # Load existing job data
        job_file = JOBS_DIR / f"{job_id}.json"
        
        if job_file.exists():
            with open(job_file, "r") as f:
                job_data = json.load(f)
        else:
            # Create new job data if file doesn't exist
            job_data = {
                "id": job_id,
                "created_at": time.time()
            }
        
        # Update fields
        job_data["status"] = status
        job_data["updated_at"] = time.time()
        
        if result is not None:
            job_data["result"] = result
        
        if error is not None:
            job_data["error"] = error
        
        # Save to file
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        
        # Update in-memory cache
        JOBS[job_id] = job_data
        
        return True
    except Exception as e:
        logger.error(f"Error updating job status: {str(e)}")
        return False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/convert', methods=['POST'])
@require_api_key
def convert_article():
    """Handle article conversion requests"""
    try:
        # Get form data
        text = request.form.get('article_text')
        
        if not text:
            return jsonify({"error": "No article text provided"}), 400
        
        # Use default values since we removed these options from the form
        tts_model = "openai"
        custom_intro = ""
        
        # Generate a unique ID for this job
        job_id = str(uuid.uuid4())
        
        # Create job metadata with simple status
        job_data = {
            "id": job_id,
            "status": "pending",
            "created_at": time.time(),
            "request": {
                "text": text,
                "tts_model": tts_model,
                "custom_intro": custom_intro
            },
            "result": None,
            "error": None
        }
        
        # Save job data
        job_file = JOBS_DIR / f"{job_id}.json"
        with open(job_file, "w") as f:
            json.dump(job_data, f)
        
        # Store job in memory
        JOBS[job_id] = job_data
        
        # Process the job (in a real app, you'd use a task queue like Celery)
        threading.Thread(target=process_podcast_job, args=(job_id, text, tts_model, custom_intro)).start()
        
        # Generate status URL for monitoring
        status_url = url_for('job_status', job_id=job_id, _external=True)
        
        # Check if the request wants JSON (API) or HTML (browser)
        # If the request Accept header prefers JSON or a specific format param is provided
        if (request.headers.get('Accept') and 'application/json' in request.headers.get('Accept')) or \
           request.args.get('format') == 'json' or request.form.get('format') == 'json':
            # Create JSON response for API clients
            json_response = {
                "success": True,
                "job_id": job_id,
                "status": "pending",
                "status_url": status_url,
                "api_status_url": url_for('api_job_status', job_id=job_id, _external=True)
            }
            
            # Log the JSON response being sent
            logger.info(f"Converting article: Returning JSON response: {json.dumps(json_response)}")
            
            return jsonify(json_response)
        else:
            # Redirect to the status page for browser clients
            logger.info(f"Converting article: Redirecting to status page: {status_url}")
            return redirect(status_url)
        
    except Exception as e:
        logger.error(f"Error creating job: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/status/<job_id>')
def job_status(job_id):
    """Show job status page"""
    try:
        # Check if job is in memory
        job_data = get_job_data(job_id)
        if not job_data:
            return render_template('error.html', error="Job not found"), 404
        
        return render_template('status.html', job=job_data)
        
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        return render_template('error.html', error=str(e)), 500

@app.route('/api/status/<job_id>')
@require_api_key
def api_job_status(job_id):
    """API endpoint to check job status"""
    try:
        # Check if job is in memory
        job_data = get_job_data(job_id)
        if not job_data:
            return jsonify({"error": "Job not found"}), 404
        
        # Build response
        response = {
            "job_id": job_id,
            "status": job_data["status"],
        }
        
        # Add additional info based on status
        if job_data["status"] == "completed" and job_data.get("result"):
            response["transcript_url"] = url_for('download_transcript', job_id=job_id)
            response["audio_url"] = url_for('download_audio', job_id=job_id)
        
        if job_data["status"] == "failed" and job_data.get("error"):
            response["error"] = job_data["error"]
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error checking job status: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/transcript/<job_id>', methods=['GET'])
@require_api_key
def download_transcript(job_id):
    """Download the transcript file for a job"""
    try:
        job_data = get_job_data(job_id)
        if not job_data:
            return jsonify({"error": "Job not found"}), 404
            
        if job_data["status"] != "completed" or not job_data.get("result"):
            return jsonify({"error": "Transcript not available"}), 400
            
        transcript_path = job_data["result"]["transcript"]
        if not os.path.exists(transcript_path):
            return jsonify({"error": "Transcript file not found"}), 404
            
        return send_file(transcript_path, as_attachment=True, download_name=f"transcript_{job_id}.txt")
    except Exception as e:
        logger.error(f"Error downloading transcript: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/download/audio/<job_id>', methods=['GET'])
@require_api_key
def download_audio(job_id):
    """Download the audio file for a job"""
    try:
        job_data = get_job_data(job_id)
        if not job_data:
            return jsonify({"error": "Job not found"}), 404
            
        if job_data["status"] != "completed" or not job_data.get("result"):
            return jsonify({"error": "Audio not available"}), 400
            
        audio_path = job_data["result"]["audio"]
        if not os.path.exists(audio_path):
            return jsonify({"error": "Audio file not found"}), 404
            
        return send_file(audio_path, as_attachment=True, download_name=f"podcast_{job_id}.mp3")
    except Exception as e:
        logger.error(f"Error downloading audio: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Main entry point
if __name__ == "__main__":
    # Load existing jobs
    load_jobs()
    
    # Run the app
    app.run(host='0.0.0.0', port=8000, debug=True)
