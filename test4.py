from __future__ import division
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
import pyaudio
import wave

# Function to read WAV file using PyAudio
def read_wav(file_path):
    CHUNK = 1024
    wf = wave.open(file_path, 'rb')
    p = pyaudio.PyAudio()

    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)

    frames = []
    data = wf.readframes(CHUNK)
    while data:
        frames.append(data)
        data = wf.readframes(CHUNK)

    stream.stop_stream()
    stream.close()
    p.terminate()

    return np.frombuffer(b''.join(frames), dtype=np.int16), wf.getframerate()

def write_wav(file_path, data, sample_rate):
    # Normalize the data to the range [-32768, 32767]
    normalized_data = (data * 32767 / np.max(np.abs(data))).astype(np.int16)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)  # Adjust this based on the number of channels in your data
    wf.setsampwidth(2)  # Adjust this based on the sample width (in bytes) of your data
    wf.setframerate(sample_rate)
    wf.writeframes(normalized_data.tobytes())
    wf.close()

# Loading wav files using PyAudio
voice_1, fs_1 = read_wav("mix_type_2_1.wav")
voice_2, fs_2 = read_wav("mix_type_2_2.wav")

# Reshaping the files to have the same size
m = min(len(voice_1), len(voice_2))
voice_1 = voice_1[:m]
voice_2 = voice_2[:m]

# Mixing data
voice = np.c_[voice_1, voice_2]
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(voice, A)

# Blind source separation using ICA
ica = FastICA()
ica.fit(X)
S_ = ica.transform(X)

# Save estimated signals to WAV files
write_wav("estimated_voice_1.wav", S_[:, 0], fs_1)
write_wav("estimated_voice_2.wav", S_[:, 1], fs_2)

# Rest of the code remains unchanged
