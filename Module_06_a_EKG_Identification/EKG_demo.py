import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# ----------------------------
# Step 1: Setup Signal
# ----------------------------
fs = 500  # Sampling frequency (Hz)
duration = 10  # seconds
t = np.linspace(0, duration, fs * duration)  # time vector

# ----------------------------
# Step 2: Simulate ECG Beat (Gaussian blobs)
# ----------------------------
def simulate_ecg_beat(t):
    ecg = np.zeros_like(t)
    p_wave = 0.1 * np.exp(-((t - 0.2)**2) / (2 * 0.01**2))
    q_wave = -0.15 * np.exp(-((t - 0.35)**2) / (2 * 0.005**2))
    r_wave = 1.0 * np.exp(-((t - 0.4)**2) / (2 * 0.002**2))
    s_wave = -0.25 * np.exp(-((t - 0.45)**2) / (2 * 0.005**2))
    t_wave = 0.35 * np.exp(-((t - 0.6)**2) / (2 * 0.02**2))
    return p_wave + q_wave + r_wave + s_wave + t_wave

# Repeat ECG beat at 1 Hz (one per second)
ecg_signal = np.zeros_like(t)
for beat_start in np.arange(0, duration, 1):
    ecg_signal += simulate_ecg_beat(t - beat_start)

# ----------------------------
# Step 3: Add Realistic Noise
# ----------------------------
np.random.seed(42)
white_noise = 0.05 * np.random.randn(len(t))
hf_noise = 0.03 * np.sin(2 * np.pi * 60 * t)
noisy_ecg = ecg_signal + white_noise + hf_noise

# ----------------------------
# Step 4: Apply Bandpass Filter (0.5–40 Hz)
# ----------------------------
def bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=500, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return lfilter(b, a, data)

filtered_ecg = bandpass_filter(noisy_ecg)

# ----------------------------
# Step 5: Extract One Beat (1s–2s window)
# ----------------------------
start = fs * 1
end = fs * 2
raw_beat = noisy_ecg[start:end]
filtered_beat = filtered_ecg[start:end]
beat_time = t[start:end]

# ----------------------------
# Step 6: Detect Fiducial Points (using peak locations)
# ----------------------------
r_idx = np.argmax(filtered_beat)
r_time = beat_time[r_idx]

q_idx = r_idx - np.argmin(filtered_beat[r_idx - int(0.05*fs):r_idx][::-1])
q_time = beat_time[q_idx]

s_idx = r_idx + np.argmin(filtered_beat[r_idx:r_idx + int(0.05*fs)])
s_time = beat_time[s_idx]

p_onset_idx = np.argmin(np.abs(beat_time - (r_time - 0.2)))
p_onset_time = beat_time[p_onset_idx]

p_peak_idx = p_onset_idx + np.argmax(filtered_beat[p_onset_idx:q_idx])
p_peak_time = beat_time[p_peak_idx]

t_onset_idx = s_idx + np.argmin(np.abs(beat_time[s_idx:] - (r_time + 0.2)))
t_peak_idx = t_onset_idx + np.argmax(filtered_beat[t_onset_idx:t_onset_idx + int(0.1 * fs)])
t_onset_time = beat_time[t_onset_idx]
t_peak_time = beat_time[t_peak_idx]

baseline_after_s = np.where(np.diff(np.sign(filtered_beat[s_idx:])))[0]
rl_idx = s_idx + baseline_after_s[0] if len(baseline_after_s) > 0 else s_idx + 10
rl_time = beat_time[rl_idx]

# ----------------------------
# Step 7: Compute Fiducial Features (all positive values)
# ----------------------------
fiducials = {
    'RQ': abs(r_time - q_time),
    'RS': abs(s_time - r_time),
    'RP': abs(r_time - p_onset_time),
    'RL': abs(r_time - rl_time),
    "RP'": abs(r_time - p_peak_time),
    'RT': abs(t_onset_time - r_time),
    "RS'": abs(r_time - s_time),
    "RT'": abs(t_peak_time - r_time),
    'P width': abs(p_peak_time - p_onset_time),
    'T width': abs(t_peak_time - t_onset_time),
    'ST': abs(t_onset_time - s_time),
    'PQ': abs(q_time - p_onset_time),
    'PT': abs(t_onset_time - p_onset_time),
    'LQ': abs(q_time - rl_time),
    "ST'": abs(t_peak_time - s_time)
}

# Normalize all features to the RR interval
rr_interval = 1.0  # assumed RR interval = 1 sec
fiducials_norm = {k: v / rr_interval for k, v in fiducials.items()}

# ----------------------------
# Step 8: Display Fiducial Features
# ----------------------------
print("Normalized Fiducial Intervals (0–1 scale, all positive):")
for k, v in fiducials_norm.items():
    print(f"{k:<6}: {v:.3f}")

# ----------------------------
# Step 9: Plot Raw and Filtered Beat
# ----------------------------
plt.figure(figsize=(14, 6))

# Raw signal
plt.subplot(2, 1, 1)
plt.plot(beat_time, raw_beat, label='Noisy ECG Beat')
plt.title('Unfiltered ECG Beat (1s–2s segment)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.legend()

# Filtered with markers
plt.subplot(2, 1, 2)
plt.plot(beat_time, filtered_beat, label='Filtered ECG Beat', color='orange')
plt.plot(r_time, filtered_beat[r_idx], 'ro', label='R')
plt.plot(q_time, filtered_beat[q_idx], 'go', label='Q')
plt.plot(s_time, filtered_beat[s_idx], 'mo', label='S')
plt.plot(p_onset_time, filtered_beat[p_onset_idx], 'c^', label='P onset')
plt.plot(p_peak_time, filtered_beat[p_peak_idx], 'co', label='P peak')
plt.plot(t_onset_time, filtered_beat[t_onset_idx], 'k^', label='T onset')
plt.plot(t_peak_time, filtered_beat[t_peak_idx], 'ko', label='T peak')
plt.axvline(x=rl_time, color='gray', linestyle='--', label='RL (baseline)')
plt.title('Filtered ECG Beat with Fiducial Points')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (mV)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()
