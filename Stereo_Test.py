import pyaudio
import struct
import wave

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
