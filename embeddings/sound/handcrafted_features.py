# Imports

from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import soundfile as sf

# Functions

def extract_audio_features(file_path):
    y, sr = sf.read(Path(file_path).as_posix())
    
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    rms_energy = librosa.feature.rms(y=y)
    
    return [
        np.mean(spectral_centroids),
        np.mean(zero_crossing_rate),
        np.mean(rms_energy)
    ]

# NRG vs. FLUX vs. CENTROID

# Variables

folder_name = "audio_files"
all_features = []
file_names = []

instrument_map = {}
instrument_map["Tp"] = "Trumpet"
instrument_map["Tbn"] = "Trombone"
instrument_map["Hn"] = "Horn"
instrument_map["BTb"] = "Bass Trombone"
instrument_map["Vc"] = "Violincello"
instrument_map["Vn"] = "Violin"
instrument_map["Va"] = "Viola"
instrument_map["Cb"] = "Contrabass"
instrument_map["Hp"] = "Harp"
instrument_map["Gtr"] = "Guitar"
instrument_map["Acc"] = "Accordian"
instrument_map["ASax"] = "Alto Sax"
instrument_map["Ob"] = "Oboe"
instrument_map["Fl"] = "Flute"
instrument_map["ClBb"] = "Clarinet"
instrument_map["Bn"] = "Bassoon"

# Scripting

file_paths = list(Path(folder_name).rglob('*'))

for file_path in tqdm(file_paths, desc="Processing files..."):
    if file_path.is_file():
        if file_path.suffix == '.wav':
            for key, val in instrument_map.items():
                if key in file_path.as_posix():
                    file_names.append(val)
            features = extract_audio_features(file_path)
            all_features.append(features)

all_features = np.array(all_features)
file_names = np.array(file_names)
np.save('processed_sound_files/hc_audio_features.npy', all_features)
np.save('processed_sound_files/hc_audio_labels.npy', file_names)