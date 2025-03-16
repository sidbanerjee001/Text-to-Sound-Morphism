from pathlib import Path
import librosa
import numpy as np
from tqdm import tqdm
from random import sample

# Process audio files via MFCC encoding and save to processed_sound_files

def load_audio_features(file_paths, max_files=None):
    features = []
    labels = []

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

    file_paths = sample(file_paths, max_files) if max_files else file_paths

    for file_path in tqdm(file_paths, desc="Processing audio files"):
        file_path = Path(file_path)

        if file_path.suffix == ".wav":
            try:
                y, sr = librosa.load(str(file_path), sr=22050)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfcc, axis=1)
                features.append(mfcc_mean)
                for key, val in instrument_map.items():
                    if key in file_path.as_posix():
                        labels.append(val)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    features = np.array(features)
    labels = np.array(labels)
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)

    return features, labels

folder_name = "../../data/audio_files" # Change this to the datapath for your audiofiles
file_paths = list(Path(folder_name).rglob('*'))

features, labels = load_audio_features(file_paths, max_files=1500)

np.save('processed_sound_files/mfcc_features.npy', features)
np.save('processed_sound_files/mfcc_labels.npy', labels)

print("Saved features and labels.")