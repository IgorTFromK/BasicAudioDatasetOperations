import librosa
import soundfile as sf
import numpy as np


def load(filename: str, sample_rate: int) -> (np.ndarray, int):
    return librosa.load(filename, sample_rate)


def write(x: np.ndarray, filename: str, sample_rate: int):
    sf.write(filename, x, sample_rate)


def resample(x: np.ndarray, src_sample_rate: int, dest_sample_rate: int) -> np.ndarray:
    return librosa.resample(x, src_sample_rate, dest_sample_rate)


def trim_audio(x: np.ndarray, duration: int, sample_rate: int) -> np.ndarray:
    out = np.zeros((duration * sample_rate))
    for i in range(duration * sample_rate):
        out[i] = x[i]
    return out


def cut_audio_chunks(x: np.ndarray, window_size: int, hop_size: int) -> list:
    chunk_list = []
    num_chunks = (x.shape[0] - window_size) // hop_size + 1
    for i in range(num_chunks):
        tmp = np.zeros(window_size)
        for j in range(window_size):
            tmp[j] = x[(i * hop_size) + j]
        chunk_list.append(tmp)
    return chunk_list


def save_wav_as_numpy_file(src_audio_file: str, dest_numpy_file: str, sample_rate: int):
    y, sr = load(src_audio_file, sample_rate)
    np.save(dest_numpy_file, y)


