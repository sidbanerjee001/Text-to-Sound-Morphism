from pathlib import Path
from matplotlib import pyplot as plt
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

def compute_avg_fft(y, target_dim=1024):
    fft_values = np.abs(librosa.stft(y, n_fft=(target_dim-1)*2))
    fft_values = np.mean(fft_values, axis=1, keepdims=True)
    fft_values = fft_values.T
    return fft_values[:min(len(fft_values), target_dim)]

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

# Function to randomly sample 80% of .wav files in a folder
def sample_wav_files(folder):
    wav_files = list(folder.glob("*.wav"))
    sample_size = max(1, len(wav_files) // 10 * 8) # at least 1 file should be sampled!
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
        
        y, sr = librosa.load(file_path, sr=44100)
        sound_data.append(y)
        max_length = max(max_length, len(y))

padded_signals = [np.pad(y, (0, max_length - len(y)), mode='constant') for y in tqdm(sound_data, desc="Padding signals...")]
fft_matrix = np.array([compute_avg_fft(y, target_dim=1024) for y in tqdm(padded_signals, desc="Computing FFTs...")])

# Save the results
np.save('processed_sound_files/sound_data.npy', fft_matrix)
np.save('processed_sound_files/sound_labels.npy', sound_labels)

print("FFT data and labels saved.")