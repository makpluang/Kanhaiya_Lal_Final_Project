
from scipy import signal
from scipy.fft import fft, fftshift
import matplotlib.pyplot as plt

window = signal.windows.hamming(51)
plt.plot(window)
plt.title("Hamming window")
plt.ylabel("Amplitude")
plt.xlabel("Samples")

plt.figure()
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.7, 0.7, -120, 0])
plt.title("Frequency response")
plt.ylabel("Normalized magnitude in [dB]")
plt.xlabel("Normalized frequency")



window = signal.windows.hann(51)
plt.plot(window)
plt.title("Hanning window")
plt.ylabel("Amplitude")
plt.xlabel("Samples")

plt.figure()
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.7, 0.7, -120, 0])
plt.title("Frequency response")
plt.ylabel("Normalized magnitude in [dB]")
plt.xlabel("Normalized frequency")

from scipy import signal
window=signal.get_window('triang', 100)
plt.plot(window)
plt.title("Triangular  window")
plt.ylabel("Amplitude")
plt.xlabel("Samples")

plt.figure()
A = fft(window, 2048) / (len(window)/2.0)
freq = np.linspace(-0.5, 0.5, len(A))
response = 20 * np.log10(np.abs(fftshift(A / abs(A).max())))
plt.plot(freq, response)
plt.axis([-0.7, 0.7, -120, 0])
plt.title("Frequency response")
plt.ylabel("Normalized magnitude in [dB]")
plt.xlabel("Normalized frequency")