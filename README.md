# Chaicast

Chaicast is a web application that transforms text articles into podcast-style audio conversations. It uses AI to convert plain text into engaging dialogues between two speakers and then generates high-quality audio using text-to-speech technology.

Chaicast Screenshot:

<img width="1440" alt="image" src="https://github.com/user-attachments/assets/e4deb6a7-0d4d-4b12-8c15-41dfce2b6b41" />


## Features

- Convert any text article into a natural-sounding podcast
- AI-powered conversation generation
- High-quality text-to-speech using OpenAI or ElevenLabs voices
- Modern, responsive web interface
- Download both audio files and transcripts
- Real-time status updates during podcast generation

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- API keys for OpenAI (required)
- API key for ElevenLabs (optional)
- Pinggy.io account for public access (optional)

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/ChaitanyaRajeev/Chaicast.git
cd Chaicast
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Create required directories**

```bash
mkdir -p audio jobs transcripts
```

5. **Set up environment variables**

Create a `.env` file in the project root with the following content:

```
OPENAI_API_KEY=your_openai_api_key
ELEVENLABS_API_KEY=your_elevenlabs_api_key_if_using
```

### Running the Application

1. **Start the Flask server**

```bash
python app.py
```

2. **Access the application**

Open your browser and navigate to:
```
http://localhost:8080
```

### Sharing Your Application (Optional)

To make your application accessible from the internet using Pinggy.io:

1. **Install Pinggy.io** (if you have a Pro account)

```bash
ssh -p 443 -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -R0:localhost:8080 YOUR_ACCESS_TOKEN@a.pinggy.io
```

Replace `YOUR_ACCESS_TOKEN` with your Pinggy.io access token.

2. **Access your application from anywhere**

Your application will be available at the URL provided by Pinggy.io.

## Usage

1. Enter or paste your article text into the input field
2. Click "Generate Podcast"
3. Wait for the processing to complete
4. Listen to your podcast directly in the browser
5. Download the audio file or transcript as needed

## Technologies Used

- **Backend**: Python, Flask
- **Frontend**: HTML, CSS, Bootstrap 5
- **AI Services**: OpenAI GPT for conversation generation, OpenAI TTS for voice synthesis
- **Alternative TTS**: ElevenLabs (optional)

## Project Structure

- `app.py` - Main Flask application
- `podcast.py` - Core podcast generation logic
- `templates/` - HTML templates for the web interface
- `audio/` - Generated audio files (not included in repository)
- `jobs/` - Job status tracking (not included in repository)
- `transcripts/` - Generated transcript files (not included in repository)

Highlevel Diagram :
![image](https://github.com/user-attachments/assets/6d563cdc-d6b3-4f8c-a89f-0081773c704d)


## Acknowledgements

- OpenAI for their powerful GPT and TTS APIs
- ElevenLabs for their realistic voice synthesis technology
- Pinggy.io for the tunneling service
