import streamlit as st
from gradio_client import Client

# Initialize Gradio Client
client = Client("stabilityai/stable-diffusion-3.5-large")

def generate_image(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps):
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

prompt = st.text_input("Enter Prompt:")
negative_prompt = st.text_input("Enter Negative Prompt:")
seed = st.number_input("Seed", value=0, step=1)
randomize_seed = st.checkbox("Randomize Seed", value=True)
width = st.slider("Width", min_value=256, max_value=2048, value=1024, step=256)
height = st.slider("Height", min_value=256, max_value=2048, value=1024, step=256)
guidance_scale = st.slider("Guidance Scale", min_value=1.0, max_value=20.0, value=4.5, step=0.5)
num_inference_steps = st.slider("Number of Inference Steps", min_value=1, max_value=100, value=40, step=1)

if st.button("Generate Image"):
    with st.spinner("Generating image..."):
        image_url = generate_image(prompt, negative_prompt, seed, randomize_seed, width, height, guidance_scale, num_inference_steps)
        if image_url:
            st.image(image_url, caption="Generated Image", use_column_width=True)
        else:
            st.error("Failed to generate image. Please try again.")
