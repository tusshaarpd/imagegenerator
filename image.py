import streamlit as st
from gradio_client import Client

# Initialize Gradio Client
client = Client("stabilityai/stable-diffusion-3.5-large")

def generate_image(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
    """
    Generates an AI-generated image using the StabilityAI Stable Diffusion model.
    
    Parameters:
        prompt (str): A description of what you want the AI to generate.
        negative_prompt (str): Things you want to exclude from the generated image.
        seed (int): A number that controls randomness. The same seed with the same settings produces the same image.
        randomize_seed (bool): If enabled, a different seed is used every time for varied results.
        width (int): The width of the generated image in pixels.
        height (int): The height of the generated image in pixels.
        guidance_scale (float): Controls how much the model follows the prompt. Higher values make it more accurate but less creative.
        num_inference_steps (int): The number of steps taken to generate the image. More steps improve quality but take longer.
    
    Returns:
        str: URL of the generated image.
    """
    result = client.predict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        randomize_seed=randomize_seed,
        width=width,
        height=height,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps,
        api_name="/infer"
    )
    return result[0]['url'] if result and 'url' in result[0] else None

# Streamlit UI
st.title("Stable Diffusion Image Generator")
st.write("Generate AI-generated images by providing a description and adjusting various parameters.")

prompt = st.text_input("Enter Prompt:", help="Describe what you want the AI to generate.")
negative_prompt = st.text_input("Enter Negative Prompt:", help="Describe what you want to exclude from the image.")
seed = st.number_input("Seed", value=0, step=1, help="Controls randomness. The same seed with the same settings produces the same image.")
randomize_seed = st.checkbox("Randomize Seed", value=True, help="If checked, a different seed is used for each generation.")
width = st.slider("Width", min_value=256, max_value=2048, value=1024, step=256, help="Width of the generated image in pixels.")
height = st.slider("Height", min_value=256, max_value=2048, value=1024, step=256, help="Height of the generated image in pixels.")
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=4.5, step=0.5, help="Higher values make the model follow the prompt more strictly but reduce creativity.")
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=100, value=40, step=1, help="More steps improve quality but take longer.")

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image_url = generate_image(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)
        if image_url:
            st.image(image_url, caption="Generated Image", use_column_width=True)
        else:
            st.error("Failed to generate image. Please try again.")
