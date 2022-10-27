#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt


# In[2]:


import librosa
from scipy.signal import kaiserord, lfilter, firwin, freqz
from pylab import figure, clf, plot, xlabel, ylabel, xlim, ylim, title, grid, axes, show


# In[3]:


Input_Dir =r'D:\CourseMaterials\Project\MachineLearning\dataset\trainset'
audio_path =  glob.glob(os.path.join(Input_Dir, '*/*.wav'))
len(audio_path)


# In[4]:


audio = []
MAX_SOUND_CLIP_DURATION=10
sample_rates = []
for path in audio_path:
    aud, sr = librosa.load(path , sr=None, duration = MAX_SOUND_CLIP_DURATION)
    sample_rates.append(sr)
    audio.append(aud)
len(audio)


# In[5]:


SAMPLE_RATE = sample_rates[0]
SAMPLE_RATE


# In[6]:


reference_path =  glob.glob(os.path.join(Input_Dir, '*/*.csv'))
len(reference_path)


# In[7]:


reference = []
for path in reference_path:
    data = pd.read_csv(path,header = None)
    reference.extend(data.to_numpy()) 
len(reference)


# In[8]:


target = []
for x,y in reference:
    target.append(y)
target


# In[9]:


def filter_audio(audio):
    sample_rate = SAMPLE_RATE 
    nyq_rate = sample_rate / 2.0
    width = 5.0/nyq_rate
    ripple_db = 60.0
    N, beta = kaiserord(ripple_db, width)
    cutoff_hz = 650
    taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    filtered_audio = lfilter(taps, 1.0, audio)
    return filtered_audio


# In[10]:


filtered_audio = []
for file in audio:
    filtered_audio.append(filter_audio(file))
len(filtered_audio)


# In[11]:


from numpy import pi, absolute, arange
sample_rate = SAMPLE_RATE
nsamples = sample_rate * MAX_SOUND_CLIP_DURATION
t = arange(nsamples) / sample_rate
nyq_rate = sample_rate / 2.0
width = 20.0/nyq_rate
ripple_db = 60.0
N, beta = kaiserord(ripple_db, width)
cutoff_hz = 650.0
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))

figure(1)
plot(taps, 'bo-', linewidth=2)
title('Filter Coefficients (%d taps)' % N)
xlim(135,230)
grid(True)

figure(2)
clf()
w, h = freqz(taps, worN=8000)
plot((w/pi)*nyq_rate, absolute(h), linewidth=2)
xlabel('Frequency (Hz)')
ylabel('Gain')
title('Frequency Response')
ylim(-0.05, 1.05)
xlim(0,1000)
grid(True)

delay = 0.5 * (N-1) / sample_rate

figure(3)
# Plot the original signal.
plot(t, audio[0])
# Plot the filtered signal, shifted to compensate for the phase delay.
plot(t-delay, filtered_audio[0], 'r-')

xlabel('t')
grid(True)


# In[12]:


import librosa.feature
dataset = []
for x in filtered_audio :
    entry = []
    y = x
    rms = librosa.feature.rms(y=y)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=SAMPLE_RATE)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)
    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE)
    to_append = f'{np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
    for value in mfcc:
        to_append += f' {np.mean(value)}'
    stuff = list(map(float, to_append.split(" ")))
    entry = entry + stuff
    dataset.append(entry)


# In[13]:


header = 'rms chroma_stft spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
for i in range(1, 21):
    header += f' mfcc{i}'
header = header.split()
#header
df = pd.DataFrame(data=dataset, columns=header)


# In[14]:


df['target'] = target
df["target"] = df["target"].apply(lambda x: 0 if x==-1 else 1)


# In[15]:


df


# In[16]:


#df.to_csv('heartsounds.csv')


# In[17]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# In[18]:


X=df.drop(columns=['target']).values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = df['target'].values


# In[242]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=69)


# In[243]:


df['target'].value_counts()


# In[244]:


X_test


# # Neural Network

# In[245]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


# In[246]:


from numpy import random

model =  Sequential()
l1 = random.randint(200,250)
l2 = random.randint(200,250)
#l3 = random.randint(0,80)

model.add(Dense(l1, activation='relu'))
model.add(Dropout(0.3))

model.add(Dense(l2,activation='relu'))
model.add(Dropout(0.3))

#model.add(Dense(l3,activation='relu'))
#model.add(Dropout(0.2))


model.add(Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',  metrics=['accuracy'])


# In[238]:


from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25)


# In[247]:


model.fit(X_train, y_train, batch_size=64, epochs = 600, callbacks=early_stop, validation_data=(X_test, y_test) )


# In[248]:


y_pred = model.predict(X_test)
y_pred = y_pred > 0.5
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


# In[249]:


from sklearn.metrics import classification_report,confusion_matrix
conf = confusion_matrix(y_test, y_pred)
conf


# In[168]:


sensitivity = conf[1,1] / (conf[1,1] + conf[1,0])
specificity = conf[0,0] / (conf[0,0] + conf[0,1])
print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")


# In[161]:


print(classification_report(y_test,y_pred))


# In[30]:


#model.save('my_model_82%.h5')


# In[ ]:




