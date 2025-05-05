import os
import json
import openai
import groq
from dotenv import load_dotenv
from deepgram import DeepgramClient, PrerecordedOptions, SpeakOptions
from apikey import OPENAIKEY, DGAPIKEY, GROQAPIKEY

os.environ['OPENAI_API_KEY'] = OPENAIKEY
os.environ['DG_API_KEY'] = DGAPIKEY
os.environ['GROQ_API_KEY'] = GROQAPIKEY

load_dotenv()
DG_API_KEY = os.getenv("DG_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not DG_API_KEY or not OPENAI_API_KEY or not GROQ_API_KEY:
    raise ValueError("Please set the DG_API_KEY and/or OPENAI_API_KEY and/or GROQ_API_KEY environment variable.")

# Initialize the APIs
deepgram = DeepgramClient(DG_API_KEY)
groq_client = groq.Groq(api_key=GROQ_API_KEY)
openai_client = openai.OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# Define the system prompt for OpenAI
system_prompt = """
You are a helpful and friendly customer service assistant for a cell phone provider.
Your goal is to help customers with issues like:
- Billing questions
- Troubleshooting their mobile devices
- Explaining data plans and features
- Activating or deactivating services
- Transferring them to appropriate departments for further assistance

Maintain a polite and professional tone in your responses. Always make the customer feel valued and heard.
"""

# Set Deepgram options for TTS and STT
text_options = PrerecordedOptions(
    model="nova-2",
    language="en",
    summarize="v2", 
    topics=True, 
    intents=True, 
    smart_format=True, 
    sentiment=True, 
)

speak_options = SpeakOptions(
    model="aura-asteria-en",
    encoding="linear16",
    container="wav"
)
def ask_groq(prompt, model="llama3-8b-8192"):
    """
    Send a prompt to the Groq API and return the response.
    """
    try:
        response = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except groq.GroqError as e:
        return f"An error occurred: {e}"
    
def ask_openai(prompt):
    """
    Send OpenAI API a prompt, returns a response back.
    """
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except openai.OpenAIError as e:
        return f"An error occurred: {e}"


def get_transcript(payload, options=text_options):
    """
    Returns a JSON of Deepgram's transcription given an audio file.
    """
    response = deepgram.listen.rest.v("1").transcribe_file(payload, options).to_json()
    return json.loads(response)


def get_topics(transcript):
    """
    Returns back a list of all unique topics in a transcript.
    """
    topics = set()  # Initialize an empty set to store unique topics

    # Traverse through the JSON structure to access topics
    for segment in transcript['results']['topics']['segments']:
        # Iterate over each topic in the current segment
        for topic in segment['topics']:
            # Add the topic to the set
            topics.add(topic['topic'])
    return topics


def get_summary(transcript):
    """
    Returns the summary of the transcript as a string.
    """
    return transcript['results']['summary']['short']


def save_speech_summary(transcript, options=speak_options):
    """
    Writes an audio summary of the transcript to disk.
    """
    s = {"text": transcript}
    filename = "output.wav"
    response = deepgram.speak.rest.v("1").save(filename, s, options)