**BachNet🎵** is a production‑ready deep learning system for generating music in the style of J.S. Bach. Trained on a corpus of 382 chorales with a multi‑layer, sequence‑to‑sequence LSTM network, it learns both melodic patterns and temporal structures from sequences of 256 notes.

From a short seed segment of a chorale, BachNet can autoregressively compose entirely new pieces. Notes are sampled from a categorical distribution, with the degree of variation controlled by a temperature parameter.

The project also incorporates a fully in‑graph TensorFlow data‑streaming pipeline, enabling efficient, on‑the‑fly creation and batching of training samples. This design keeps the CPU busy preparing data while the GPU remains fully utilized for model training, maximizing both throughput and performance. Project GitHUB: [github.com/hoom4n/BachNet](https://github.com/hoom4n/BachNet)
