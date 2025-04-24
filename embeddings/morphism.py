# Naive morphism--finding closest Euclidean point in sound latent space.

import numpy as np

text_data = np.load("text/embedding_outputs/umap10k_word2vec_3d.npz", allow_pickle=True)
text_vectors_3d = text_data["vectors"]
w2v_words = text_data["words"]

text_data = np.load("sound/embedding_outputs/tsne_samples_3d.npz", allow_pickle=True)
sound_vectors_3d = text_data["vectors"]
instrument_labels = text_data["instruments"]

sentence = "life is reserved for family"
sentence = sentence.split()
sentence_vectors = []

for word in sentence:
    for i in range(len(w2v_words)):
        if word == w2v_words[i]:
            sentence_vectors.append(text_vectors_3d[i])

sound_points = []
for word_vec in sentence_vectors:
    distances = []
    for i in range(len(sound_vectors_3d)):
        distances.append(np.linalg.norm(sound_vectors_3d[i] - word_vec))
    sound_points.append(instrument_labels[np.argmin(distances)])

# Right now, the points in the latent space of sound are labeled by Instrument. I'll have to retrain the AE on filepaths to be able to 
# construct a series of samples to play (the ultimate goal being sentence => sound/[music?])
print(sound_points)