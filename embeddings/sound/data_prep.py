from pathlib import Path
import numpy as np
from tqdm import tqdm
import librosa
from random import sample

# Define the folder containing audio files
folder_name = "../../data/audio_files"
file_paths = list(Path(folder_name).rglob('*'))

def is_lowest_level_folder(folder):
    return not any(item.is_dir() for item in folder.iterdir())

# Instrument mapping dictionary
instrument_map = {
    "Tp": "Trumpet", "Tbn": "Trombone", "Hn": "Horn", "BTb": "Bass Trombone",
    "Vc": "Violincello", "Vn": "Violin", "Va": "Viola", "Cb": "Contrabass",
    "Hp": "Harp", "Gtr": "Guitar", "Acc": "Accordian", "ASax": "Alto Sax",
    "Ob": "Oboe", "Fl": "Flute", "ClBb": "Clarinet", "Bn": "Bassoon"
}

def compute_avg_fft(y, window_size=2048, hop_length=512, target_dim=1024):
    # stft_matrix = librosa.stft(y, n_fft=window_size, hop_length=hop_length)
    # magnitude_spectrum = np.abs(stft_matrix)
    # mean_spectrum = np.mean(magnitude_spectrum, axis=1)

    # mean_spectrum = np.interp(
    #     np.linspace(0, len(mean_spectrum), target_dim),
    #     np.arange(len(mean_spectrum)),
    #     mean_spectrum
    # )
    
    # return mean_spectrum
    
    # Compute STFT (Short-Time Fourier Transform)
    stft_matrix = librosa.stft(y, n_fft=window_size, hop_length=hop_length)
    
    # Convert to magnitude spectrum
    magnitude_spectrum = np.abs(stft_matrix)
    
    # Compute mean FFT across time
    mean_spectrum = np.mean(magnitude_spectrum, axis=1)
    
    # Ensure exactly 1024 dimensions (drop last element if needed)
    avg_fft = mean_spectrum[:1024]
    
    return avg_fft

sound_data = []
sound_labels = []
max_length = 0

def get_lowest_level_folders(parent_folder):
    parent_folder = Path(parent_folder)
    lowest_level_folders = []
    for folder in parent_folder.rglob("*"):
        if folder.is_dir() and is_lowest_level_folder(folder):
            lowest_level_folders.append(folder)
    return lowest_level_folders

# Function to randomly sample 75% of .wav files in a folder
def sample_wav_files(folder):
    wav_files = list(folder.glob("*.wav"))
    sample_size = max(1, len(wav_files) // 10 * 9) # at least 1 file should be sampled!
    return sample(wav_files, sample_size)

lowest_level_folders = get_lowest_level_folders(folder_name)

sampled_files = []
for folder in lowest_level_folders:
    sampled_files.extend(sample_wav_files(folder))

# Process each file
for file_path in tqdm(sampled_files, desc="Processing sound files..."):
    if file_path.is_file() and file_path.suffix == '.wav':
        # Determine the instrument label
        for key, val in instrument_map.items():
            if key in file_path.as_posix():
                sound_labels.append(val)
                break
        
        y, sr = librosa.load(file_path, sr=None)
        sound_data.append(y)
        max_length = max(max_length, len(y))

padded_signals = [np.pad(y, (0, max_length - len(y)), mode='constant') for y in tqdm(sound_data, desc="Padding signals...")]
fft_matrix = np.array([compute_avg_fft(y, target_dim=1024) for y in tqdm(padded_signals, desc="Computing FFTs...")])

# Save the results
np.save('processed_sound_files/sound_data.npy', fft_matrix)
np.save('processed_sound_files/sound_labels.npy', sound_labels)

print("FFT data and labels saved.")