import streamlit as st
# from audio_recorder_streamlit import audio_recorder  # Commented out for now
import numpy as np
from tensorflow.keras.models import load_model
import librosa
import matplotlib.pyplot as plt
import librosa.display

SAMPLE_RATE = 44100
N_MELS = 128
max_time_steps = 109
# Load your audio classification model
model = load_model("audio_classifier18.h5")

# Function to preprocess audio
def preprocess_audio(audio, target_sr=44100, target_shape=(128, 109)):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Ensure all spectrograms have the same width (time steps)
    if mel_spectrogram.shape[1] < max_time_steps:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_time_steps - mel_spectrogram.shape[1])), mode='constant')
    else:
        mel_spectrogram = mel_spectrogram[:, :max_time_steps]

    return mel_spectrogram

# Function to make predictions
def predict(processed_data):
    # Make sure there are no non-finite values in the processed_data
    if not np.all(np.isfinite(processed_data)):
        st.error("Non-finite values detected in the processed audio data. Unable to make predictions.")
        return None

    # Make predictions using your model
    prediction = model.predict(processed_data.reshape(1, 128, 109, 1))

    # Assuming binary classification with softmax output
    class_labels = ["IT IS A FAKE VOICE", "IT IS A REAL VOICE"]
    predicted_class = class_labels[int(prediction[0][0] > 0.5)]

    return predicted_class

# Streamlit app
st.set_page_config(
    page_title="Deep Fake Detection",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="collapsed",
)
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
backgground-image: url("https://img.freepik.com/free-vector/sinuous-background_23-2148487911.jpg?size=626&ext=jpg&ga=GA1.1.2008272138.1697155200&semt=ais")
background-size: cover;
}
[data-testid="stHeader"] {
background-color: rgba(0,0,0,0);
}
</style>
"""
st.title("Deep Fake Detection")
st.header("Upload Audio")

# Commented out recording for now
# Record audio using audio_recorder_streamlit
# audio_bytes = audio_recorder()

# Option to upload an audio file
uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "flac"])

if uploaded_file:
    # Display the uploaded file details
    st.audio(uploaded_file, format='audio/wav')
    audio_np = librosa.load(uploaded_file, sr=SAMPLE_RATE)[0]

    # Convert audio_np to float32 and normalize to the range [-1, 1]
    audio_np = audio_np.astype(np.float32) / np.iinfo(np.int16).max

    # Check for non-finite values
    if not np.all(np.isfinite(audio_np)):
        st.error("Non-finite values detected in the recorded audio. Unable to proceed.")
    else:
        # Preprocess the audio
        processed_data = preprocess_audio(audio_np)

        # Display file details
        st.write("Audio details:")
        st.write(f"Sample Rate: {SAMPLE_RATE} Hz")
        st.write(f"Duration: {len(audio_np) / SAMPLE_RATE:.2f} seconds")

        # Plot the waveform and spectrogram side by side
        st.subheader("Waveform and Spectrogram:")
        col1, col2 = st.columns(2)

        with col1:
            # Plot the waveform
            fig_waveform, ax_waveform = plt.subplots(figsize=(10, 4))
            librosa.display.waveshow(audio_np, sr=SAMPLE_RATE, ax=ax_waveform)
            st.pyplot(fig_waveform)

        with col2:
            # Plot the spectrogram
            fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(processed_data[:, :], sr=SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax_spec)
            colorbar = plt.colorbar(format='%+2.0f dB', ax=ax_spec, mappable=img)
            st.pyplot(fig_spec)

        # Add a Predict button
        if st.button("Predict"):
            # Make predictions
            predicted_class = predict(processed_data)

            # Display predictions with larger font size
            if predicted_class is not None:
                st.markdown("<div style='font-size: 24px; text-align: left; color: white;'>Prediction: {}</div>".format(predicted_class), unsafe_allow_html=True)
