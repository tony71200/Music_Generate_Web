# Music_Generate_Web
AudioCraft provides the code and models for MusicGen, [a simple and controllable model for music generation][arxiv].
MusicGen is a single stage auto-regressive Transformer model trained over a 32kHz
<a href="https://github.com/facebookresearch/encodec">EnCodec tokenizer</a> with 4 codebooks sampled at 50 Hz.
Unlike existing methods like [MusicLM](https://arxiv.org/abs/2301.11325), MusicGen doesn't require
a self-supervised semantic representation, and it generates all 4 codebooks in one pass. By introducing
a small delay between the codebooks, we show we can predict them in parallel, thus having only 50 auto-regressive
steps per second of audio.
Check out our [sample page][musicgen_samples] or test the available demo!

<a target="_blank" href="https://ai.honu.io/red/musicgen-colab">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
<a target="_blank" href="https://huggingface.co/spaces/facebook/MusicGen">
  <img src="https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm.svg" alt="Open in HugginFace"/>
</a>
<br>

We use 20K hours of licensed music to train MusicGen. Specifically, we rely on an internal dataset
of 10K high-quality music tracks, and on the ShutterStock and Pond5 music data.


## Installation
AudioCraft requires Python 3.9 or upper, PyTorch 2.1.0, Anaconda (Anaconda Mini). To install AudioCraft, you can run the following:
- Create environment in conda
```shell
conda create --name music python=3.9
# Activate env
conda activate music
```
In the environment, install the required packages
```shell
# Best to make sure you have torch installed first, in particular before installing xformers.
# Don't run this if you already have PyTorch installed.
python -m pip install 'torch==2.1.0'
# You might need the following before trying to install the packages
python -m pip install setuptools wheel
# Then proceed to one of the following
python -m pip install -U audiocraft  # stable release
python -m pip install -U git+https://git@github.com/facebookresearch/audiocraft#egg=audiocraft  # bleeding edge
python -m pip install -e .  # or if you cloned the repo locally (mandatory if you want to train).
pip install streamlit
```

To run the web app, you can use the following command
```shell
streamlit run app.py
```