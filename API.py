import google.generativeai as genai
import os
import gradio as gr
from elevenlabs.client import ElevenLabs
from dotenv import load_dotenv

# 1.
load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

if not GOOGLE_API_KEY or not ELEVENLABS_API_KEY:
    raise ValueError("API keys not found. Please set them in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)
eleven_client = ElevenLabs(api_key=ELEVENLABS_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')


#2. CORE LOGIC FUNCTIONS
def chat_with_gemini(full_prompt):
    """
    Generates a text response from Gemini based on the full prompt.
    """
    
    generation_config = genai.types.GenerationConfig(max_output_tokens=20)
    
    response = model.generate_content(full_prompt, generation_config=generation_config)
    
    if not response.text or response.text.strip() == "":
        return "I'm sorry, I couldn't generate a response for that."
    
    return response.text

def text_to_speech_file(text):
    """
    Converts text to speech, saves it to a file, and returns the filepath.
    """
    try:
        # ElevenLabs API to convert text to audio
        audio_generator = eleven_client.text_to_speech.convert(
            text=text,
            voice_id="Xb7hH8MSUJpSbSDYk0k2", 
            model_id="eleven_turbo_v2",
        )

        complete_audio_bytes = b"".join(audio_generator)

        
        output_filepath = "response.mp3"
        
        with open(output_filepath, "wb") as f:
            f.write(complete_audio_bytes)            
        return output_filepath
    except Exception as e:
        print(f"Error converting text to speech: {e}")
        return None


# 3. GRADIO FUNCTION - WITH MEMORY)
def chatbot_response(user_prompt, history):
    """
    This is the main function that Gradio calls. It orchestrates the entire
    chatbot logic, including managing conversation history.
    """
    if history is None:
        history = []

    context = "\n".join(history[-4:])
    
    instruction = "Your output must be a complete sentence, under 15 words, and very concise."

    prompt_with_context = f"{context}\nUser: {user_prompt}\n\n{instruction}\nAI:"
    
    gemini_text = chat_with_gemini(prompt_with_context)
    
    audio_file_path = text_to_speech_file(gemini_text)
    
    history.append(f"User: {user_prompt}")
    history.append(f"AI: {gemini_text}")
    
    return audio_file_path, history


# 4. GRADIO
iface = gr.Interface(
    fn=chatbot_response,
    inputs=[
        gr.Textbox(label="Your Message"), 
        gr.State() 
    ],
    outputs=[
        gr.Audio(label="Chatbot Response", autoplay=True), 
        gr.State()  
    ],
    title="Voice Chatbot",
    description="""
    This is a voice-enabled chatbot powered by Google's Gemini and ElevenLabs. 
    Type a message and click 'Submit' to hear a spoken response. 
    The chatbot remembers the last few turns of your conversation.
    """,
    flagging_mode="never"
)



if __name__ == "__main__":
    # This starts the web server and makes the app accessible
    iface.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860))
    )














