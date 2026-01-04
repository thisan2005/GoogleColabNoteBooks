import librosa
import numpy as np
import pandas as pd
import os

def extract_mfcc_features(audio_dir, output_csv):
    features = []
    for idx, filename in enumerate(os.listdir(audio_dir)):
        if filename.endswith('.mp3'):
            filepath = os.path.join(audio_dir, filename)
            try:
                y, sr = librosa.load(filepath)
                mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
                mfcc_mean = np.mean(mfcc, axis=1)  # Mean across time axis
                row = [idx, filename] + mfcc_mean.tolist()
                features.append(row)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    if features:
        columns = ['song_index', 'file_name'] + [f'mfcc_{i}' for i in range(20)]
        df = pd.DataFrame(features, columns=columns)
        df.to_csv(output_csv, index=False)
        print(f"Features extracted and saved to {output_csv}")
    else:
        print("No audio files found.")

if __name__ == "__main__":
    audio_dir = 'dataset/audio'
    output_csv = 'dataset/features_large.csv'
    extract_mfcc_features(audio_dir, output_csv)