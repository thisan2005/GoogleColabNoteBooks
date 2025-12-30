# Imports
import os
import glob
import numpy as np
import soundfile as sf
import openl3
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Config
DATASET_DIR = "dataset"  # structure: dataset/<label>/*.wav
CONTENT_TYPE = "music"   # "music" or "env"
EMBED_MODEL = "music"    # openl3 model params (openl3 handles model selection via input args)
EMBED_INPUT_REPR = "mel256"  # or 'linear'
EMBED_DIM = 512  # 512 or 6144 depending on model; openl3 returns shape accordingly

def load_audio_mono(path):
    audio, sr = sf.read(path, dtype='float32')
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio, sr

def file_embeddings(path, content_type=CONTENT_TYPE, input_repr=EMBED_INPUT_REPR, embedding_size=512):
    audio, sr = load_audio_mono(path)
    emb, ts = openl3.get_audio_embedding(audio, sr,
                                        content_type=content_type,
                                        input_repr=input_repr,
                                        embedding_size=embedding_size,
                                        center=True,
                                        hop_size=1.0)  # hop_size in seconds; adjust if needed
    # emb shape: (n_frames, embedding_dim) → aggregate to single vector
    return np.mean(emb, axis=0)

def build_embeddings(dataset_dir=DATASET_DIR):
    X = []
    y = []
    classes = sorted([d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir,d))])
    for label in classes:
        files = glob.glob(os.path.join(dataset_dir, label, "*"))
        for f in tqdm(files, desc=f"Processing {label}"):
            try:
                vec = file_embeddings(f)
                X.append(vec)
                y.append(label)
            except Exception as e:
                print("Failed", f, e)
    X = np.vstack(X)
    y = np.array(y)
    return X, y

# Extract embeddings (this can take time)
X, y = build_embeddings()

# Save embeddings for reuse
np.save("embeddings.npy", X)
np.save("labels.npy", y)

# Train / evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

clf = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(clf, "music_classifier.joblib")

# Predict helper
def predict_file(path):
    vec = file_embeddings(path)
    model = joblib.load("music_classifier.joblib")
    return model.predict([vec])[0], model.predict_proba([vec])

# Example usage:
# label, probs = predict_file("some_song_clip.wav")
# print(label, probs)