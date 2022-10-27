import tensorflow
from tensorflow.keras.models import load_model
import librosa
from scipy.signal import kaiserord, lfilter, firwin
import numpy as np
from joblib import load

MAX_SOUND_CLIP = 10
OUTCOMES = ['No murmur detected', 'Murmur Detected']

def filter_aud(audio, sar):
    nyq_rate = sar / 2.0
    width = 5.0/nyq_rate
    ripple_db = 60.0
    N, beta = kaiserord(ripple_db, width)
    cutoff_hz = 650
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    filtered_audio = lfilter(taps, 1.0, audio)
    return filtered_audio


def predict(file):
    aud, sr = librosa.load(file, sr=None, duration= MAX_SOUND_CLIP)
    features = []
    y = filter_aud(aud, sar=sr)
    rms = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for value in mfcc:
        to_append += f' {np.mean(value)}'
    features = [list(map(float, to_append.split(" ")))]
    scalar = load('std_scaler.bin')
    features = scalar.transform(features)
    model = load_model("my_model.h5")
    predicted = model.predict(features)
    certainty = [max(pred) for pred in predicted]
    predictions = [np.argmax(np.array(list(map(int,pred == max(pred))))) for pred in predicted]
    output = {}
    output['prediction'] = bool(predictions[0])
    output['audio_sample_rate'] = sr
    predictions = [OUTCOMES[p] for p in predictions]
    output['alt_predicion']= predictions[0]
    output['certainty'] = float(certainty[0])
    output['audio_array'] = aud.tolist()
    return output
