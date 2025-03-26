"""
Standalone Text to Podcast Converter

A completely self-contained script to convert text into podcast audio.
"""

#!/usr/bin/env python

import os
import sys
import uuid
import argparse
import json
import tempfile
import time
import logging
import re
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("text_to_podcast")

logger.info("Starting Standalone Text to Podcast Converter...")

# Check and install required packages
required_packages = ["openai", "requests", "pydub", "elevenlabs", "python-dotenv"]
missing_packages = []

logger.info("Checking for required packages...")
for package in required_packages:
    try:
        __import__(package)
        logger.info(f"√ {package} is installed")
    except ImportError:
        missing_packages.append(package)
        logger.warning(f"✗ {package} is NOT installed")

if missing_packages:
    logger.info(f"Installing missing packages: {', '.join(missing_packages)}")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing_packages)
        logger.info("All required packages installed successfully")
    except Exception as e:
        logger.error(f"Failed to install required packages: {e}")
        logger.error("You may need to install them manually using: pip install " + " ".join(missing_packages))
        sys.exit(1)

# Now import the packages
try:
    import openai
    import requests
    from pydub import AudioSegment
    import elevenlabs
    from dotenv import load_dotenv
except ImportError as e:
    logger.error(f"Failed to import required packages even after installation: {e}")
    logger.error("Please try installing them manually: pip install openai requests pydub elevenlabs python-dotenv")
    sys.exit(1)

# Load environment variables from .env file
logger.info("Loading environment variables from .env file...")
load_dotenv()

# Get API keys from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")

# Check if API keys are available
if not openai_api_key:
    logger.warning("OPENAI_API_KEY not found in environment variables or .env file")
if not gemini_api_key:
    logger.warning("GEMINI_API_KEY not found in environment variables or .env file")
if not elevenlabs_api_key:
    logger.warning("ELEVENLABS_API_KEY not found in environment variables or .env file")

# Set ffmpeg path for pydub if on Mac
if sys.platform == 'darwin':
    logger.info("Setting ffmpeg path for macOS...")
    os.environ["PATH"] += os.pathsep + "/opt/homebrew/bin"
elif sys.platform == 'linux':
    logger.info("Setting ffmpeg path for Linux...")
    os.environ["PATH"] += os.pathsep + "/usr/bin"

# Check for ffmpeg
try:
    logger.info("Checking for ffmpeg...")
    import subprocess
    result = subprocess.run(["which", "ffmpeg"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode == 0:
        logger.info(f"ffmpeg found at: {result.stdout.strip()}")
    else:
        logger.warning("ffmpeg not found! You may need to install it manually.")
        logger.warning("  - macOS: brew install ffmpeg")
        logger.warning("  - Linux: sudo apt-get install ffmpeg")
        logger.warning("  - Windows: download from https://ffmpeg.org/download.html")
except Exception as e:
    logger.warning(f"Couldn't check for ffmpeg: {e}")

# Check for ffmpeg again and provide a more helpful error if not found
try:
    # Try to load a sample file to test if ffmpeg is working
    logger.info("Testing pydub/ffmpeg setup...")
    from pydub.utils import mediainfo
    ffmpeg_works = True
except Exception as e:
    ffmpeg_works = False
    logger.warning(f"FFmpeg might not be properly configured: {e}")
    logger.warning("Audio processing might fail unless ffmpeg is properly installed.")

class SimpleTextToSpeech:
    """A simplified text-to-speech converter that supports OpenAI and ElevenLabs"""
    
    def __init__(self, api_key, model="openai"):
        self.model = model
        self.api_key = api_key
        logger.info(f"Initializing TTS with {model} model")
        
        if model == "openai":
            openai.api_key = api_key
        elif model == "elevenlabs":
            elevenlabs.set_api_key(api_key)
    
    def convert_to_speech(self, text, output_file):
        """Convert text to speech and save to output_file"""
        logger.info(f"Starting text-to-speech conversion with {self.model}...")
        
        if self.model == "openai":
            return self._convert_with_openai(text, output_file)
        elif self.model == "elevenlabs":
            return self._convert_with_elevenlabs(text, output_file)
        else:
            logger.error(f"Unsupported TTS model: {self.model}")
            raise ValueError(f"Unsupported TTS model: {self.model}")
    
    def _convert_with_openai(self, text, output_file):
        """Use OpenAI's TTS API to convert text to speech"""
        # Split text by speaker tags
        segments = []
        current_pos = 0
        
        person1_voice = "alloy"
        person2_voice = "shimmer"
        
        logger.info("Processing transcript to extract speaker segments...")
        
        # Process each part of text with appropriate voice
        while current_pos < len(text):
            p1_start = text.find("<Person1>", current_pos)
            p2_start = text.find("<Person2>", current_pos)
            
            if p1_start == -1 and p2_start == -1:
                break
            
            if (p1_start < p2_start and p1_start != -1) or p2_start == -1:
                # Person1 segment
                start = p1_start + len("<Person1>")
                end = text.find("</Person1>", start)
                if end == -1:
                    end = len(text)
                
                content = text[start:end].strip()
                # Remove any "Person 1:" or "Alex:" prefixes that might have been added
                content = re.sub(r'^(Person 1:|Alex:)\s*', '', content)
                segments.append({"text": content, "voice": person1_voice})
                current_pos = end + len("</Person1>")
                
            elif (p2_start < p1_start and p2_start != -1) or p1_start == -1:
                # Person2 segment
                start = p2_start + len("<Person2>")
                end = text.find("</Person2>", start)
                if end == -1:
                    end = len(text)
                
                content = text[start:end].strip()
                # Remove any "Person 2:" or "Taylor:" prefixes that might have been added
                content = re.sub(r'^(Person 2:|Taylor:)\s*', '', content)
                segments.append({"text": content, "voice": person2_voice})
                current_pos = end + len("</Person2>")
        
        logger.info(f"Found {len(segments)} speaker segments")
        
        # Create audio segments
        audio_files = []
        for i, segment in enumerate(segments):
            temp_file = f"temp_audio_{i}.mp3"
            logger.info(f"Generating audio for segment {i+1}/{len(segments)} with voice {segment['voice']}")
            
            try:
                # Call OpenAI TTS API
                response = openai.audio.speech.create(
                    model="tts-1",
                    voice=segment["voice"],
                    input=segment["text"]
                )
                
                # Save to temp file
                response.stream_to_file(temp_file)
                audio_files.append(temp_file)
            except Exception as e:
                logger.error(f"Error generating speech for segment {i+1}: {str(e)}")
                raise
                
        logger.info("Combining audio segments...")
        
        # Combine all segments
        combined = AudioSegment.empty()
        for file in audio_files:
            try:
                segment = AudioSegment.from_mp3(file)
                combined += segment
                
                # Short pause between segments (500ms)
                combined += AudioSegment.silent(duration=500)
            except Exception as e:
                logger.error(f"Error combining segment {file}: {str(e)}")
                
        logger.info(f"Saving combined audio to {output_file}")
        combined.export(output_file, format="mp3")
        
        # Clean up temp files
        for file in audio_files:
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Error removing temp file {file}: {str(e)}")
                
        return output_file

    def _convert_with_elevenlabs(self, text, output_file):
        """Use ElevenLabs API to convert text to speech"""
        
        # If elevenlabs API key is not set, skip this
        if not self.api_key:
            logger.error("ElevenLabs API key not provided")
            raise ValueError("ElevenLabs API key is required for this TTS model")
        
        # Split text by speaker tags
        segments = []
        current_pos = 0
        
        # ElevenLabs voices - you can change these to your preferred voices
        person1_voice = "Antoni"  # Male voice
        person2_voice = "Rachel"  # Female voice
        
        logger.info("Processing transcript to extract speaker segments...")
        
        # Process each part of text with appropriate voice
        while current_pos < len(text):
            p1_start = text.find("<Person1>", current_pos)
            p2_start = text.find("<Person2>", current_pos)
            
            if p1_start == -1 and p2_start == -1:
                break
            
            if (p1_start < p2_start and p1_start != -1) or p2_start == -1:
                # Person1 segment
                start = p1_start + len("<Person1>")
                end = text.find("</Person1>", start)
                if end == -1:
                    end = len(text)
                
                content = text[start:end].strip()
                segments.append({"text": content, "voice": person1_voice})
                current_pos = end + len("</Person1>")
                
            elif (p2_start < p1_start and p2_start != -1) or p1_start == -1:
                # Person2 segment
                start = p2_start + len("<Person2>")
                end = text.find("</Person2>", start)
                if end == -1:
                    end = len(text)
                
                content = text[start:end].strip()
                segments.append({"text": content, "voice": person2_voice})
                current_pos = end + len("</Person2>")
        
        logger.info(f"Found {len(segments)} speaker segments")
        
        # Create audio segments
        audio_files = []
        for i, segment in enumerate(segments):
            temp_file = f"temp_audio_{i}.mp3"
            logger.info(f"Generating audio for segment {i+1}/{len(segments)} with voice {segment['voice']}")
            
            try:
                # Generate audio with ElevenLabs
                audio = elevenlabs.generate(
                    text=segment["text"],
                    voice=segment["voice"],
                    model="eleven_monolingual_v1"
                )
                
                # Save to file
                with open(temp_file, "wb") as f:
                    f.write(audio)
                
                audio_files.append(temp_file)
            except Exception as e:
                logger.error(f"Error generating speech for segment {i+1}: {str(e)}")
                raise
                
        logger.info("Combining audio segments...")
        
        # Combine all segments
        combined = AudioSegment.empty()
        for file in audio_files:
            try:
                segment = AudioSegment.from_mp3(file)
                combined += segment
                
                # Short pause between segments (500ms)
                combined += AudioSegment.silent(duration=500)
            except Exception as e:
                logger.error(f"Error combining segment {file}: {str(e)}")
                
        logger.info(f"Saving combined audio to {output_file}")
        combined.export(output_file, format="mp3")
        
        # Clean up temp files
        for file in audio_files:
            try:
                os.remove(file)
            except Exception as e:
                logger.warning(f"Error removing temp file {file}: {str(e)}")
                
        return output_file

class SimpleContentGenerator:
    """Generates conversational content from input text using OpenAI"""
    
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = api_key
        logger.info("Initializing content generator")
    
    def generate_conversation(self, input_text, custom_intro=""):
        """Generate a conversation transcript from input text"""
        logger.info("Generating conversational transcript...")
        
        # Create system prompt with custom_intro
        system_prompt = f"""
        You are a professional podcast scriptwriter. Convert the given content into a natural, engaging conversation 
        between two hosts named Alex and Taylor.

        Rules:
        1. Start with {custom_intro or "a brief introduction to the topic"}
        2. Create a natural back-and-forth dialogue that covers the main points
        3. Add conversational elements, questions, and reactions
        4. Make it sound like a real conversation, not a robotic reading
        5. End with a brief conclusion
        6. Format using <Person1>...</Person1> for Alex and <Person2>...</Person2> for Taylor tags exactly
        7. Use the term "Chaicast" or mention "Your Personal Generative AI Podcast developed by Chai"
        8. DO NOT include "Person 1:" or "Person 2:" or "Alex:" or "Taylor:" in the actual spoken text

        DO NOT:
        - Add any unnecessary branding
        - Invent facts not present in the original text
        """
        
        logger.info(f"Input text length: {len(input_text)} characters")
        
        try:
            logger.info("Calling OpenAI API to generate conversation...")
            response = openai.chat.completions.create(
                model="gpt-4-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": input_text}
                ],
                temperature=0.7,
                max_tokens=4000
            )
            
            transcript = response.choices[0].message.content
            logger.info(f"Generated transcript length: {len(transcript)} characters")
            return transcript
        except Exception as e:
            logger.error(f"Error generating conversation: {e}")
            logger.warning("Falling back to simplified format...")
            # Simplified fallback format
            return f"<Person1>{input_text}</Person1>"


def create_output_directories():
    """Create directories for output files"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Create directories
    transcript_dir = os.path.join(script_dir, "output", "transcripts")
    audio_dir = os.path.join(script_dir, "output", "audio")
    
    logger.info(f"Creating output directories: {transcript_dir} and {audio_dir}")
    os.makedirs(transcript_dir, exist_ok=True)
    os.makedirs(audio_dir, exist_ok=True)
    
    return transcript_dir, audio_dir


def convert_text_to_podcast(text, tts_model="openai", custom_intro="", openai_api_key=None, elevenlabs_api_key=None):
    """
    Convert text to a podcast audio file.
    
    Args:
        text (str): The text content to convert
        tts_model (str): The text-to-speech model to use ('openai' or 'elevenlabs')
        custom_intro (str): Custom introduction for the podcast
        openai_api_key (str): OpenAI API key
        elevenlabs_api_key (str): ElevenLabs API key
        
    Returns:
        tuple: Paths to the generated transcript and audio files
    """
    try:
        # Get API keys from arguments or environment
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        elevenlabs_api_key = elevenlabs_api_key or os.getenv("ELEVENLABS_API_KEY")
        
        # Check required API keys
        if not openai_api_key:
            logger.error("OpenAI API key not provided")
            return None, None
            
        if tts_model == "elevenlabs" and not elevenlabs_api_key:
            logger.error("ElevenLabs API key not provided but ElevenLabs TTS selected")
            return None, None
        
        # Create output directories
        output_dir = Path("./output")
        transcript_dir = output_dir / "transcripts"
        audio_dir = output_dir / "audio"
        
        # Generate unique filenames using UUID
        timestamp = str(uuid.uuid4())
        transcript_file = transcript_dir / f"transcript_{timestamp}.txt"
        audio_file = audio_dir / f"podcast_{timestamp}.mp3"
        
        # Generate conversation from input text
        logger.info("Generating conversational content...")
        content_gen = SimpleContentGenerator(openai_api_key)
        conversation = content_gen.generate_conversation(text, custom_intro)
        
        # Save transcript
        logger.info(f"Saving transcript to {transcript_file}")
        transcript_file.parent.mkdir(exist_ok=True)
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(conversation)
        
        # Convert transcript to speech
        logger.info(f"Converting transcript to audio using {tts_model}...")
        if tts_model == "openai":
            # Use OpenAI for TTS
            tts_engine = SimpleTextToSpeech(openai_api_key, model="openai")
        elif tts_model == "elevenlabs":
            # Use ElevenLabs for TTS
            tts_engine = SimpleTextToSpeech(elevenlabs_api_key, model="elevenlabs")
        else:
            logger.error(f"Unsupported TTS model: {tts_model}")
            return transcript_file, None
        
        # Perform the actual conversion
        tts_engine.convert_to_speech(conversation, str(audio_file))
        
        logger.info(f"Podcast created successfully: {audio_file}")
        return str(transcript_file), str(audio_file)
        
    except Exception as e:
        logger.error(f"Error converting text to podcast: {str(e)}")
        return None, None


def main():
    """Main entry point for the script"""
    try:
        # Set up argument parser
        parser = argparse.ArgumentParser(description="Convert text to podcast audio")
        parser.add_argument("--openai-key", help="OpenAI API key")
        parser.add_argument("--elevenlabs-key", help="ElevenLabs API key")
        parser.add_argument("--tts", choices=["openai", "elevenlabs"], default="openai", help="TTS model to use (default: openai)")
        parser.add_argument("--intro", default="", help="Custom introduction for the podcast")
        parser.add_argument("--use-hardcoded", action="store_true", help="Use hardcoded article text")
        parser.add_argument("--input-file", help="Input text file to convert")
        parser.add_argument("--text", help="Direct text input to convert")
        parser.add_argument("--debug", action="store_true", help="Enable debug logging")
        args = parser.parse_args()
        
        # Configure debug logging if requested
        if args.debug:
            logger.setLevel(logging.DEBUG)
            logger.debug("Debug logging enabled")
        
        # Handle API keys - first check command line args, then environment variables
        openai_api_key = args.openai_key or os.getenv("OPENAI_API_KEY") or ""
        elevenlabs_api_key = args.elevenlabs_key or os.getenv("ELEVENLABS_API_KEY") or ""
        
        # Check required API keys
        if not openai_api_key:
            logger.error("No OpenAI API key provided. Exiting.")
            return
        
        if args.tts == "elevenlabs" and not elevenlabs_api_key:
            logger.warning("ElevenLabs API key not provided but ElevenLabs TTS selected")
            elevenlabs_api_key = input("Enter your ElevenLabs API key: ")
            if not elevenlabs_api_key.strip():
                logger.error("No ElevenLabs API key provided but ElevenLabs TTS selected. Exiting.")
                return
        
        # Determine the input text source
        input_text = None
        
        # Use hardcoded article option
        if args.use_hardcoded:
            # PASTE YOUR ARTICLE HERE BETWEEN THE TRIPLE QUOTES
            hardcoded_article = """
elcome to the wild, wonderful — and sometimes worrisome — world of product management in 2025 and beyond. If you've been riding the tech roller coaster for a while, you know how dizzying the ups and downs can be. Economic fluctuations, AI revolutions, and ever-changing user expectations have all made their mark — and there's no stopping that momentum now. So what does it take for a product manager (PM) to thrive (not just survive) in this new era? Let's dive in with some actionable insights and fresh perspectives to help you stand out from the crowd and keep your product strategy on solid ground — no matter what new shift comes your way.

1. Business Acumen Above All Else
Product managers today are more than just backlog managers or feature gatekeepers. With budgets tighter and scrutiny higher, every PM must be deeply aware of how their decisions drive (or drain) the bottom line. If you can speak the language of finance, you immediately become more credible to executive teams. If you can interpret revenue metrics as easily as you do conversion metrics, your influence grows tenfold.

How to Build It:

Be curious about balance sheets and P&Ls. Ask your finance team to walk you through the company's main revenue streams and cost centers.
Connect product metrics to business impact. Show how improvements in user experience correlate with an uptick in paid conversions, renewed subscriptions, or retention.
2. The "Full-Stack" PM Mindset
Ever heard of a full-stack developer? Now imagine a full-stack product manager. While a "pure" PM may have once thrived by focusing on planning and prioritization alone, 2025 demands a wider skill set. Smaller teams, constrained budgets, and cross-disciplinary challenges require you to blend business strategy, technical know-how, and design sensibilities.

Where to Expand:

Technical Literacy: You don't need to be a senior engineer, but you should be comfortable reading code snippets or understanding the nuances of machine learning models your product might use.
Design Appreciation: Know enough about UX to articulate what good design looks like. Having a design vocabulary helps you collaborate seamlessly with your creatives.
Data Analysis: Get hands-on with data tools. Try setting up your own SQL queries or dabbling with AI-driven analytics platforms.

            """
            
            input_text = hardcoded_article
            logger.info(f"Using hardcoded article: {len(hardcoded_article)} characters")
        
        # Check for direct text input
        elif args.text:
            input_text = args.text
            logger.info(f"Using text from command line argument: {len(args.text)} characters")
        
        # Check for input file
        elif args.input_file:
            try:
                with open(args.input_file, "r", encoding="utf-8") as f:
                    input_text = f.read()
                logger.info(f"Using text from file {args.input_file}: {len(input_text)} characters")
            except Exception as e:
                logger.error(f"Error reading input file: {e}")
                return
        
        # If no input is provided, prompt user
        else:
            logger.info("No input text provided. Please enter your text below (Ctrl+D to finish):")
            try:
                input_text = sys.stdin.read().strip()
                if not input_text:
                    logger.error("No input text provided. Exiting.")
                    return
            except KeyboardInterrupt:
                logger.info("\nCancelled by user. Exiting.")
                return
        
        # Custom introduction (modify as needed)
        custom_intro = args.intro or "Welcome to today's podcast episode"
        
        # Convert text to podcast
        transcript, audio = convert_text_to_podcast(
            input_text,
            args.tts,
            custom_intro,
            openai_api_key,
            elevenlabs_api_key
        )
        
        if transcript and audio:
            logger.info(f"Conversion successful. Files saved to:")
            logger.info(f"  - Transcript: {transcript}")
            logger.info(f"  - Audio: {audio}")
        else:
            logger.error("Conversion failed. Please check the logs for details.")
            
    except KeyboardInterrupt:
        logger.info("\nCancelled by user. Exiting.")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if "--debug" in sys.argv:
            raise

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
