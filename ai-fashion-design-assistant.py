# ============================
# 🪄 AI Fashion Design Generator (Voice + Text + Auto Description)
# ============================

# Step 1: Install libraries
!pip install diffusers transformers accelerate safetensors torch gradio gtts SpeechRecognition pydub

# Step 2: Import libraries
import gradio as gr
from diffusers import StableDiffusionPipeline
import torch
from gtts import gTTS
import tempfile
import speech_recognition as sr

# Step 3: Load the model
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Step 4: Replace this part with improved version
def generate_fashion(prompt_text, style, jewelry, age_group, prompt_voice):
    spoken_prompt = ""
    if prompt_voice:
        recognizer = sr.Recognizer()
        with sr.AudioFile(prompt_voice) as source:
            audio = recognizer.record(source)
        try:
            spoken_prompt = recognizer.recognize_google(audio)
        except:
            spoken_prompt = ""
    
    final_input = spoken_prompt if spoken_prompt else prompt_text

    if not final_input:
        return None, None, "⚠ Please type or speak your outfit idea!", "Please describe your outfit idea to begin."

    base_prompt = (
        f"A {age_group} wearing a {style.lower()} {final_input}, "
        f"featuring different models, patterns, and backgrounds, "
        f"with {jewelry.lower()} jewelry, ultra-realistic 8k resolution, "
        f"professional fashion photoshoot lighting, Vogue style."
    )

    images = []
    for seed in [123, 456, 789]:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
        image = pipe(base_prompt, num_inference_steps=30, guidance_scale=8, generator=generator).images[0]
        images.append(image)

    text_description = (
        f"Here are your AI-generated fashion designs based on your instructions. "
        f"This look represents a {style.lower()} style for a {age_group}. "
        f"The outfit idea — {final_input} — is styled with {jewelry.lower()} jewelry. "
        f"Each design highlights unique fabric textures, poses, and lighting — "
        f"perfect for inspiration or digital portfolio use."
    )

    tts = gTTS(text_description)
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(temp_audio.name)

    return images, temp_audio.name, text_description, "🎨 Your designs are ready! AI styled them as per your fashion idea."

# Step 5: Gradio UI (same as before)
with gr.Blocks() as demo:
    gr.Markdown("## 👗 AI Fashion Design Generator — Speak or Type Your Outfit Idea")

    with gr.Row():
        prompt_text = gr.Textbox(label="🧵 Type your outfit idea", placeholder="e.g. red silk saree, royal blue gown")
        prompt_voice = gr.Audio(label="🎤 Or speak your outfit idea", type="filepath")

    with gr.Row():
        style = gr.Radio(["Traditional", "Modern", "Fusion"], label="Style Type", value="Traditional")
        jewelry = gr.Radio(["None", "Light", "Gold", "Silver", "Diamond", "Kundan"], label="Jewelry Option", value="None")
        age_group = gr.Radio(["Woman", "Man", "Girl", "Boy", "Child"], label="Age Group", value="Woman")

    btn = gr.Button("🎨 Generate Fashion Designs")

    with gr.Row():
        gallery = gr.Gallery(label="✨ AI Fashion Designs (Different Models)", columns=3, height="auto")
        voice_output = gr.Audio(label="🔊 Voice Description")
    
    with gr.Row():
        description_box = gr.Textbox(label="📝 AI Text Description", interactive=False, lines=3)
        result_text = gr.Textbox(label="🪞 Status Message", interactive=False)

    btn.click(
        fn=generate_fashion,
        inputs=[prompt_text, style, jewelry, age_group, prompt_voice],
        outputs=[gallery, voice_output, description_box, result_text]
    )

demo.launch()