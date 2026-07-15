# Text-to-speech generation
# Author: Mudit Airan

# ============ STEP 3: GENERATE PODCAST AUDIO ============
# Input:  json_path   (path to a script JSON file produced by llm_processor.py's
#                       save_script(), i.e. {"ticker": ..., "script": ..., ...})
#         output_dir  (folder to write the mp3 into)
# Output: audio_path  (path to the generated ~5 minute mp3 file)

import json
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def load_script(json_path):
    """Read the script JSON file written by llm_processor.py's save_script()."""
    with open(json_path, "r") as f:
        return json.load(f)


def generate_audio(json_path, output_dir="output", voice="onyx", model="gpt-4o-mini-tts"):
    data = load_script(json_path)
    ticker = data["ticker"]
    topic = data["topic"]
    script = data["script"]

    os.makedirs(output_dir, exist_ok=True)
    # Keyed by ticker+topic (matching save_script's filename), not ticker
    # alone — otherwise two episodes for the same ticker but different
    # topics would silently overwrite each other's audio file.
    audio_path = os.path.join(output_dir, f"{ticker}_{topic}_podcast.mp3")

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=script,
    )
    response.stream_to_file(audio_path)

    return audio_path



if __name__ == "__main__":
    audio_path = generate_audio("output/AAPL_technology_script.json")
    print(f"Saved podcast audio to: {audio_path}")
