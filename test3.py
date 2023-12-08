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

# Loading wav files using PyAudio
voice_1, fs_1 = read_wav("mix_type_2_1.wav")
voice_2, fs_2 = read_wav("mix_type_2_2.wav")

# Reshaping the files to have the same size
m = min(len(voice_1), len(voice_2))
voice_1 = voice_1[:m]
voice_2 = voice_2[:m]

# Rest of the code remains unchanged
# plotting time domain representation of signal
figure_1 = plt.figure("Original Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, voice_1)
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of voice_2")
plt.plot(np.arange(m)/fs_2, voice_2)
plt.xlabel("Time")
plt.ylabel("Signal")

# mix data
voice = np.c_[voice_1, voice_2]
A = np.array([[1, 0.5], [0.5, 1]])
X = np.dot(voice, A)

# plotting time domain representation of mixed signal
figure_2 = plt.figure("Mixed Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of mixed voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, X[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of mixed voice_2")
plt.plot(np.arange(m)/fs_2, X[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")

# blind source separation using ICA
ica = FastICA()
print( "Training the ICA decomposer .....")
t_start = time.time()
ica.fit(X)
t_stop = time.time() - t_start
print ("Training Complete; took %f seconds" % (t_stop))
# get the estimated sources
S_ = ica.transform(X)
# get the estimated mixing matrix
A_ = ica.mixing_
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# plotting time domain representation of estimated signal
figure_3 = plt.figure("Estimated Signal")
plt.subplot(2, 1, 1)
plt.title("Time Domain Representation of estimated voice_1")
plt.xlabel("Time")
plt.ylabel("Signal")
plt.plot(np.arange(m)/fs_1, S_[:, 0])
plt.subplot(2, 1, 2)
plt.title("Time Domain Representation of estimated voice_2")
plt.plot(np.arange(m)/fs_2, S_[:, 1])
plt.xlabel("Time")
plt.ylabel("Signal")

plt.show()