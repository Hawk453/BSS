import pyaudio
import struct
import wave
import numpy as np
from sklearn.decomposition import FastICA

def open_wave_file(filename, channels=1):
    wf = wave.open(filename, 'w')
    wf.setnchannels(channels)
    wf.setsampwidth(WIDTH)
    wf.setframerate(RATE)
    return wf

def open_audio_stream():
    return p.open(format=pyaudio.paInt16,
                  channels=CHANNELS,
                  rate=RATE,
                  input=True,
                  output=False)

def close_resources(stream, *wave_files):
    stream.stop_stream()
    stream.close()
    p.terminate()

    for wf in wave_files:
        wf.close()

def record_to_wave_files(stream, left_wf, right_wf, duration):
    print(f'* Recording for {duration:.3f} seconds')

    for _ in range(int(RATE / BLOCKLEN * duration)):
        binary_input_data = stream.read(BLOCKLEN, exception_on_overflow=False)
        input_tuple = struct.unpack('h' * CHANNELS * BLOCKLEN, binary_input_data)

        left_samples = [input_tuple[n] for n in range(0, CHANNELS * BLOCKLEN, CHANNELS)]
        right_samples = [input_tuple[n + 1] for n in range(0, CHANNELS * BLOCKLEN, CHANNELS)]

        left_binary_output_data = struct.pack('h' * BLOCKLEN, *left_samples)
        right_binary_output_data = struct.pack('h' * BLOCKLEN, *right_samples)

        left_wf.writeframes(left_binary_output_data)
        right_wf.writeframes(right_binary_output_data)

    print('* Done')

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
    normalized_data = (data * 32767 / np.max(np.abs(data))).astype(np.int16)

    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(sample_rate)
    wf.writeframes(normalized_data.tobytes())
    wf.close()

# Audio parameters
RATE = 48000
CHANNELS = 2
BLOCKLEN = 1024
WIDTH = 2
RECORD_SECONDS = 4

# File names for left and right channels
left_filename = 'left_channel.wav'
right_filename = 'right_channel.wav'

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open wave files for left and right channels
left_wf = open_wave_file(left_filename)
right_wf = open_wave_file(right_filename)

# Open audio stream
stream = open_audio_stream()

# Record to wave files
record_to_wave_files(stream, left_wf, right_wf, RECORD_SECONDS)

# Close resources
close_resources(stream, left_wf, right_wf)

# Load mixed audio data
voice_1, fs_1 = read_wav("left_channel.wav")
voice_2, fs_2 = read_wav("right_channel.wav")

# Reshape the files to have the same size
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
