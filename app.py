import streamlit as st
import tempfile
import os
import subprocess
from pydub import AudioSegment
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import math
from tensorflow import keras

# ---------------------
# Config / caching
# ---------------------
st.set_page_config(page_title="Music Genre Prediction", layout="wide")

@st.cache_resource(show_spinner=False)
def load_model(path="genre_classifier.h5"):
    return keras.models.load_model(path)

model = load_model("genre_classifier.h5")

GENRE_DICT = {
    0: "disco", 1: "pop", 2: "classical", 3: "metal", 4: "rock",
    5: "blues", 6: "hiphop", 7: "reggae", 8: "country", 9: "jazz"
}

CONFIDENCE_THRESHOLD = 0.5  # 50%

# ---------------------
# Helper functions
# ---------------------
def convert_mp3_to_wav(input_path, out_path="converted.wav", sr=22050):
    """
    Convert any input audio to mono WAV with fixed sample rate and 16-bit PCM.
    Requires ffmpeg in PATH.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ac", "1",                 # mono
        "-ar", str(sr),             # sample rate
        "-sample_fmt", "s16",       # 16-bit PCM
        out_path
    ]
    subprocess.call(cmd)
    return out_path

def trim_first_n_seconds_wav(input_wav, out_wav="final.wav", duration_ms=30000):
    sound = AudioSegment.from_wav(input_wav)
    if len(sound) > duration_ms:
        sound = sound[:duration_ms]
    sound.export(out_wav, format="wav")
    return out_wav

def extract_mfcc_array(path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, max_frames=130):
    signal, _ = librosa.load(path, sr=sr, mono=True)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    mfcc = mfcc.T  # shape: (frames, n_mfcc)
    # pad or trim to max_frames
    if mfcc.shape[0] < max_frames:
        pad_width = max_frames - mfcc.shape[0]
        mfcc = np.pad(mfcc, ((0, pad_width), (0,0)), mode='constant')
    else:
        mfcc = mfcc[:max_frames, :]
    return mfcc

def predict_genre_from_mfcc(mfcc_array):
    X = mfcc_array[np.newaxis, ..., np.newaxis].astype(np.float32)
    preds = model.predict(X, verbose=0)[0]
    top_idx = np.argsort(preds)[-3:][::-1]
    return preds, top_idx

def plot_waveform(path, sr=22050):
    y, sr = librosa.load(path, sr=sr, mono=True)
    fig, ax = plt.subplots(figsize=(8, 3))
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set_title("Waveform")
    ax.set_xlabel("")
    ax.set_ylabel("Amplitude")
    plt.tight_layout()
    return fig

def plot_spectrogram(path, sr=22050, n_fft=2048, hop_length=512):
    y, sr = librosa.load(path, sr=sr, mono=True)
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    S_db = librosa.amplitude_to_db(S, ref=np.max)
    fig, ax = plt.subplots(figsize=(8, 3))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz', ax=ax)
    ax.set_title("Spectrogram (dB)")
    plt.colorbar(img, ax=ax, format="%+2.0f dB")
    plt.tight_layout()
    return fig

# ---------------------
# UI Layout
# ---------------------
st.title("Music Genre Prediction")
st.markdown("Upload a `.wav` or `.mp3` file. MP3s are converted to WAV for prediction.")

with st.sidebar:
    st.header("Options")
    show_waveform = st.checkbox("Show waveform", value=True)
    show_spectrogram = st.checkbox("Show spectrogram", value=True)
    trim_seconds = st.number_input("Trim length (seconds)", min_value=5, max_value=60, value=30, step=5)
    sr = st.selectbox("Sample rate (Hz)", options=[22050, 44100], index=0)
    st.divider()
    st.caption("Model loaded from `genre_classifier.h5`")

col1, col2 = st.columns([1, 1])

# File uploader
uploaded_file = st.file_uploader("Upload audio file (.wav, .mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Write to temp file
    tmp_dir = tempfile.mkdtemp()
    input_path = os.path.join(tmp_dir, uploaded_file.name)
    with open(input_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Convert/normalize
    try:
        with st.spinner("Converting / normalizing audio..."):
            converted = os.path.join(tmp_dir, "converted.wav")
            convert_mp3_to_wav(input_path, converted, sr=sr)
            final_wav = os.path.join(tmp_dir, "final.wav")
            trim_ms = int(trim_seconds * 1000)
            final_wav = trim_first_n_seconds_wav(converted, out_wav=final_wav, duration_ms=trim_ms)
    except Exception as e:
        st.error(f"Audio conversion failed: {e}")
        st.stop()

    # Audio player
    st.audio(final_wav)

    # Visuals
    if show_waveform:
        fig_w = plot_waveform(final_wav, sr=sr)
        col1.pyplot(fig_w)
    if show_spectrogram:
        fig_s = plot_spectrogram(final_wav, sr=sr)
        col2.pyplot(fig_s)

    # Extract MFCCs and predict
    with st.spinner("Extracting features and predicting..."):
        mfcc = extract_mfcc_array(final_wav, sr=sr, n_mfcc=13, n_fft=2048, hop_length=512, max_frames=130)
        preds, top_idx = predict_genre_from_mfcc(mfcc)

    # Show predictions with confidence threshold
    st.markdown("### Top Predictions")
    cols = st.columns(3)
    for i, c in enumerate(cols):
        idx = top_idx[i]
        genre = GENRE_DICT.get(int(idx), str(idx))
        prob = float(preds[idx])
        c.markdown(f"**{i+1}. {genre}**")
        c.write(f"{prob*100:.2f}%")
        c.progress(min(max(prob, 0.0), 1.0))

    # Warning if uncertain
    top_prob = float(preds[top_idx[0]])
    if top_prob < CONFIDENCE_THRESHOLD:
        st.warning(f"⚠ Model is uncertain about this prediction (top confidence: {top_prob*100:.2f}%)")
    else:
        st.success(f"Prediction seems confident ✅ (top confidence: {top_prob*100:.2f}%)")

    # Optional: full probabilities
    with st.expander("Show full probabilities"):
        prob_table = {GENRE_DICT[i]: float(preds[i])*100 for i in range(len(preds))}
        st.table(prob_table)
else:
    st.info("Upload a file to get started.")
