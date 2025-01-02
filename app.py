from audiocraft.models import MusicGen
import streamlit as st
import torch
import torchaudio
import os
import numpy as np
import base64

@st.cache_resource
def load_model():
    model = MusicGen.get_pretrained("facebook/musicgen-small")
    return model

def generate_music_tensor(description, duration: int):
    model = load_model()
    model.set_generation_params(
        use_sampling=True,
        top_k=250,
        duration=duration
    )

    output = model.generate(
        descriptions= [description],
        progress= True,
        return_tokens=True,
    
    )
    return output[0]

def save_audio(samples: torch.Tensor):
    """Renders an audio player for the given audio samples and saves them to a local directory.

    Args:
        samples (torch.Tensor): a Tensor of decoded audio samples
            with shapes [B, C, T] or [C, T]
        sample_rate (int): sample rate audio should be displayed with.
        save_path (str): path to the directory where audio should be saved.
    """
    sample_rate = 32000
    save_path = "./audio_output/"
    os.makedirs(save_path, exist_ok=True)
    assert samples.dim() == 2 or samples.dim() == 3

    samples = samples.detach().cpu()
    if samples.dim() == 2:
        samples = samples[None, ...]

    for idx, audio in enumerate(samples):
        audio_path = os.path.join(save_path, f"audio_{idx}.wav")
        torchaudio.save(audio_path, audio, sample_rate)

def get_binary_file_downloader_html(bin_file, file_label='File'):
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    return href

st.set_page_config(
    page_title= "Music Generation",
    page_icon= ":music:",
)

def main():
    st.title("Music Generation: 🎵")

    with st.expander("See explaination: "):
        st.write("This is a simple music generation model. It uses a pre-trained model to generate music.")
    
    text_area = st.text_area("Enter a text to generate music: ", height= 100)
    time_slider = st.slider("Select time duration (In Seconds) ", 10, 120, 30)

    if st.button("Generate Music") and text_area and time_slider:
        st.json({
            "Your Description": text_area,
            "Time Duration": time_slider
        })

        st.subheader("Generated Music")
        music_tensor = generate_music_tensor(text_area, time_slider)
        save_music_file = save_audio(music_tensor)
        audio_filepath = 'audio_output/audio_0.wav'
        audio_file = open(audio_filepath, 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes)
        st.markdown(get_binary_file_downloader_html(audio_filepath, 'Audio'), unsafe_allow_html=True)

if __name__ == "__main__":
    main()