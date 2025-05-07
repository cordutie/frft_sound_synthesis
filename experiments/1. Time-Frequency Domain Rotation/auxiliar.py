import math
import numpy as np
import torch
from torch_frft.frft_module  import frft  as frft_continuous
from torch_frft.dfrft_module import dfrft as frft_discrete
import matplotlib.pyplot as plt
import librosa 
from IPython.display import Audio, display, HTML

frft_type = "continuous"

if frft_type == "continuous":
    frft = frft_continuous
elif frft_type == "discrete":
    frft = frft_discrete

# Make a function that makes sinusoid torch tensor with a given frequency, length in seconds and sampling rate
def make_sinusoid(frequency, length, sampling_rate):
    time = torch.linspace(0, length, int(sampling_rate*length))
    return torch.sin(2*math.pi*frequency*time)

def make_triangle_wave(frequency, length, sampling_rate):
    time = torch.linspace(0, length, int(sampling_rate*length))
    return 2 * torch.acos(torch.cos(2*math.pi*frequency*time))

def make_square_wave(frequency, length, sampling_rate):
    time = torch.linspace(0, length, int(sampling_rate*length))
    return torch.sign(torch.sin(2*math.pi*frequency*time))

def make_sawtooth_wave(frequency, length, sampling_rate):
    time = torch.linspace(0, length, int(sampling_rate*length))
    return 2 * (time * frequency - torch.floor(0.5 + time * frequency))

# make function that load sounds using librosa into torch tensor
def load_sound(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    return torch.tensor(y), sr

# make function that takes tensor and plays it as a sound file using Audio and Display
def play_sound(tensor, fs=44100):
    display(Audio(tensor.numpy(), rate=fs))

def plot_sound(tensor):
    plt.plot(tensor.numpy())
    plt.show()

def get_segments(tensor, segment_length, hop_length, window_type="hann"):
    # Get the window function
    if window_type == "hann":
        window = torch.hann_window(segment_length)
    elif window_type == "hamming":
        window = torch.hamming_window(segment_length)
    elif window_type == "blackman":
        window = torch.blackman_window(segment_length)
    elif window_type == "rectangular":
        window = torch.ones(segment_length)
    else:
        raise ValueError("Invalid window type")
    
    device = tensor.device
    window = window.to(device)

    # Get the number of segments
    n_segments = (len(tensor) - segment_length) // hop_length + 1
    
    # Create the segments
    segments = torch.zeros(n_segments, segment_length)
    for i in range(n_segments):
        segments[i] = tensor[i*hop_length:i*hop_length+segment_length] * window
    
    return segments

def overlap_add(segments, hop_length, window_type="hann"):
    # Get the window function
    if window_type == "hann":
        window = torch.hann_window(segments.shape[1])
    elif window_type == "hamming":
        window = torch.hamming_window(segments.shape[1])
    elif window_type == "blackman":
        window = torch.blackman_window(segments.shape[1])
    elif window_type == "rectangular":
        window = torch.ones(segments.shape[1])
    else:
        raise ValueError("Invalid window type")
    
    device = segments.device
    window = window.to(device)

    # Get the number of segments
    n_segments = segments.shape[0]
    # dtype of the segments
    dtype = segments.dtype

    # Create the output tensor
    output_length = (n_segments-1)*hop_length + segments.shape[1]

    #make zeros output with dtype
    output = torch.zeros(output_length, dtype=dtype).to(device)
    
    for i in range(n_segments):
        output[i*hop_length:i*hop_length+segments.shape[1]] += segments[i] * window
    
    return output

def frft_convolver(segments_1, segments_2, angles):
    # Get the number of segments
    n_segments = segments_1.shape[0]
    
    # Get the segment length
    segment_length = segments_1.shape[1]
    
    # Create the output tensor
    output = torch.zeros(n_segments, segment_length)
    
    for i in range(n_segments):
        # Compute the fractional Fourier transform of the segment
        frft_1 = frft(segments_1[i], angles[i])
        frft_2 = frft(segments_2[i], angles[i])
        
        # Multiply the two transforms
        frft_product = frft_1 * frft_2
        
        # Compute the inverse fractional Fourier transform
        output[i] = frft(frft_product, -0.5)
    
    return output

def filter_freq_response(center_freq, q_value, window_size, fs=44100):
    """
    Compute the frequency response of a filter using only NumPy.

    Parameters:
    center_freq (float): The center frequency of the filter in Hz.
    q_value (float): The Q factor of the filter.
    window_size (int): The size of the filter window.

    Returns:
    tuple: Frequencies and their corresponding magnitude response.
    """
    if window_size % 2 == 0:
        t = np.arange(-1*(window_size // 2), window_size // 2) / fs
    else:
        t = np.arange(-1*(window_size // 2), window_size // 2 + 1) / fs

    # Generate the impulse response for a bandpass filter (Gaussian approximation)
    center_freq = center_freq.numpy()
    q_value     = q_value.numpy()
    bandwidth   = center_freq / q_value
    impulse_response = np.exp(-0.5 * ((t * bandwidth) ** 2)) * np.cos(2 * np.pi * center_freq * t)

    # Apply a window (Hann window for smoothness)
    hann_window       = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(window_size) / (window_size - 1))
    impulse_response *= hann_window

    # Compute the frequency response via FFT
    freq_response = np.fft.fft(impulse_response, n=window_size)

    # Magnitude response
    magnitude_response = np.abs(freq_response)

    return torch.tensor(magnitude_response)

def frft_filter(tensor, angle, center_freq, q_value, fs=44100):
    segment_length = tensor.shape[0]

    # Compute the filter kernel
    kernel = filter_freq_response(center_freq, q_value, segment_length, fs)
    
    # Compute the fractional Fourier transform of the segment
    frft_segment = frft(tensor, angle)

    # Apply the filter
    frft_segment_filtered = frft_segment * kernel
    
    # Compute the inverse fractional Fourier transform
    output = frft(frft_segment_filtered, -1*angle)
    
    return output

def frft_filter_automatization(tensor, angles_start_end, center_freq_start_end, q_value_start_end, window_size, hop_length, complex_part, fs=44100, window_type_input="rectangular", window_type_output="hann"):
    segments = get_segments(tensor, window_size, hop_length, window_type_input)
    n_segments = segments.shape[0]
    angles_start, angles_end           = angles_start_end
    center_freq_start, center_freq_end = center_freq_start_end
    center_freq_start, center_freq_end = torch.tensor(center_freq_start), torch.tensor(center_freq_end)
    q_value_start, q_value_end         = q_value_start_end
    angles       = torch.linspace(angles_start, angles_end, n_segments)
    center_freqs = torch.linspace(torch.log2(center_freq_start), torch.log2(center_freq_end), n_segments)**2
    q_values     = torch.linspace(q_value_start, q_value_end, n_segments)
    segnments_filtered = []
    for i in range(n_segments):
        segment     = segments[i]
        angle       = angles[i]
        center_freq = center_freqs[i]
        q_value     = q_values[i]
        local = frft_filter(segment, angle, center_freq, q_value, fs)
        if complex_part =="real_part":
            local = torch.real(local)
        elif complex_part =="imaginary_part":
            local = torch.imag(local)
        else:
            raise ValueError("Invalid complex part")
        segnments_filtered.append(local)
    segnments_filtered = torch.stack(segnments_filtered)
    output = overlap_add(segnments_filtered, hop_length, window_type_output)
    return output

def frft_windowed(tensor, windows_size, hop_length, angle_start_end, complex_part, window_type_input="rectangular", window_type_output="hann"):
    segments = get_segments(tensor, windows_size, hop_length, window_type_input)
    segments_transformed = []
    n_segments = len(segments)
    if type(angle_start_end) == list:
        angle_start, angle_end = angle_start_end
        angles     = torch.linspace(angle_start, angle_end, n_segments)
        angles = angles.to(tensor.device)
    else:
        angles = torch.ones(n_segments) * angle_start_end
        angles = angles.to(tensor.device)
    for i in range(n_segments):
        segment, angle = segments[i], angles[i]
        segment = segment.to(tensor.device)
        angle   = angle.to(tensor.device)
        local = frft(segment, angle)
        if complex_part=="real_part":
            local = torch.real(local)
        elif complex_part=="imaginary_part":
            local = torch.imag(local)
        elif complex_part=="full":
            local = local # lol
        else:
            raise Warning("Complex part not recognized so the full complex signal is used.")
        segments_transformed.append(local)
    segments_transformed = torch.stack(segments_transformed)
    output = overlap_add(segments_transformed, hop_length, window_type_output)    
    return output

def create_audio_grid(sound_lists, rows_size, cols_size, sampling_rate=44100, transpose=False):
    if transpose:
        # Transpose the sound_lists matrix
        sound_lists = list(map(list, zip(*sound_lists)))
    
    # Check the matrix dimensions match rows_size and cols_size
    if len(sound_lists) != rows_size or len(sound_lists[0]) != cols_size:
        raise ValueError("The dimensions of sound_lists must match rows_size and cols_size.")
    
    html = "<table>"
    for row in range(rows_size):
        html += "<tr>"
        for col in range(cols_size):
            name, sound = sound_lists[row][col]
            audio = Audio(sound.numpy(), rate=sampling_rate)
            audio_html = audio._repr_html_()  # Generates HTML representation for audio
            html += f"<td>{name}<br>{audio_html}</td>"
        html += "</tr>"
    html += "</table>"
    display(HTML(html))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import soundfile as sf
import subprocess
import os

def create_spectrum_video(signal, sample_rate, output_video_path, figsize=(14, 10)):
    duration = len(signal) / sample_rate
    frame_size = 4096
    hop_size = 1024

    freq_bins = np.fft.rfftfreq(frame_size, d=1/sample_rate)

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')
    ax.grid(True, color='gray', linestyle='-', linewidth=0.5)
    line, = ax.plot(freq_bins, np.zeros_like(freq_bins), color='orange')
    ax.set_xlim(freq_bins[2], freq_bins[-1])
    ax.set_ylim(-125, 40)
    ax.set_xscale('log')
    tick_frequencies = [125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    ax.set_xticks(tick_frequencies)
    ax.set_xticklabels(tick_frequencies)
    ax.set_xlabel('Frequency (Hz)', color='white')
    ax.set_ylabel('Amplitude (dB)', color='white')
    ax.tick_params(axis='both', colors='white')
    hann_window = np.hanning(frame_size)

    def update_spectrum(frame):
        start = frame * hop_size
        end = start + frame_size
        if end > len(signal):
            return line,
        windowed_signal = signal[start:end] * hann_window
        spectrum = np.abs(np.fft.rfft(windowed_signal))
        spectrum_db = 20 * np.log10(spectrum + 1e-10)
        line.set_ydata(spectrum_db)
        return line,

    ani = animation.FuncAnimation(
        fig, update_spectrum, frames=np.arange(0, len(signal) // hop_size),
        interval=1000 * hop_size / sample_rate, blit=True
    )

    temp_audio_file = output_video_path+'temp_audio.wav'
    temp_video_file = output_video_path+'temp_video.mp4'
    
    # Save the audio and video files
    sf.write(temp_audio_file, signal, sample_rate)
    ani.save(temp_video_file, writer='ffmpeg', fps=30)
    plt.close(fig)

    if os.path.exists(output_video_path):
        os.remove(output_video_path)

    # Combine video and audio using ffmpeg
    command = [
        'ffmpeg',
        '-i', temp_video_file,
        '-i', temp_audio_file,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        output_video_path
    ]
    subprocess.run(command, check=True)

    # Clean up temporary files
    if os.path.exists(temp_audio_file):
        os.remove(temp_audio_file)
    if os.path.exists(temp_video_file):
        os.remove(temp_video_file)

# # Example usage
# sr = 44100
# t = np.linspace(0, 1, sr)
# signal = np.sin(2 * np.pi * (220 + 5000 * t) * t) + np.sin(2 * np.pi * (5000 - 4500 * t) * t)
# create_spectrum_video(signal, sr, 'spectrum_video_with_audio.mp4', figsize=(14, 10))