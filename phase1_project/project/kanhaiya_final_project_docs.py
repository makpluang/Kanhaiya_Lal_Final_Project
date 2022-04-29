

colab_requirements = [
    "pip install librosa",
    "pip install noisereduce",
    "pip install soundfile",

]

import sys, subprocess

def run_subprocess_command(cmd):

    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    for line in process.stdout:
        print(line.decode().strip())

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    for i in colab_requirements:
        run_subprocess_command(i)

import IPython
from scipy.io import wavfile
import noisereduce as nr
import soundfile as sf
from noisereduce.generate_noise import band_limited_noise
import matplotlib.pyplot as plt
import urllib.request
import numpy as np
import io


from scipy.io import wavfile
rate, data = wavfile.read('/content/drive/MyDrive/Kanhaiya_Final_Project/kanhaiya.wav')

data = data
rate=rate

!pip install pydub

from pydub import AudioSegment
sound = AudioSegment.from_wav("/content/drive/MyDrive/Kanhaiya_Final_Project/kanhaiya.wav")
sound = sound.set_channels(1)
sound.export("/content/drive/MyDrive/Kanhaiya_Final_Project/path.wav", format="wav")

wav_loc = "/content/drive/MyDrive/Kanhaiya_Final_Project/path.wav"
rate, data = wavfile.read(wav_loc)

IPython.display.Audio(data=data, rate=rate)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(data)

noise_len = 2 
noise = band_limited_noise(min_freq=2000, max_freq = 18000, samples=len(data), samplerate=rate)*10
noise_clip = noise[:rate*noise_len]
audio_clip_band_limited = data+noise

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(audio_clip_band_limited)

IPython.display.Audio(data=audio_clip_band_limited, rate=rate)

reduced_noise = nr.reduce_noise(y = audio_clip_band_limited, sr=rate, n_std_thresh_stationary=1.5,stationary=True)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(reduced_noise)

IPython.display.Audio(data=reduced_noise, rate=rate)

reduced_noise = nr.reduce_noise(y = audio_clip_band_limited, sr=rate, thresh_n_mult_nonstationary=2,stationary=False)

fig, ax = plt.subplots(figsize=(20,3))
ax.plot(reduced_noise)

IPython.display.Audio(data=reduced_noise, rate=rate)