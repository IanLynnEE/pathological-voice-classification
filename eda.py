import os

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def age_distributions(path: str) -> None:
    df = pd.read_csv(path)

    # Distribution of ages for different sex.
    plt.figure(figsize=(16, 9))
    sns.kdeplot(df, x='Age', hue='Sex', palette='crest')
    plt.legend(labels=('Female', 'Male'), title='Sex')
    plt.title('Age Distribution (Male and Female)')
    plt.xlabel('Ages')
    plt.tight_layout()
    plt.savefig('figures/age_sex.png')
    plt.clf()

    disease_label = ['Phonotrauma', 'Incomplete Glottic Closure', 'Vocal Palsy', 'Neoplasm', 'Normal']
    disease_label.reverse()

    # Distribution of ages for different class without independent normalization.
    sns.kdeplot(data=df, x='Age', hue='Disease category', palette='crest')
    plt.legend(labels=disease_label, title='Category')
    plt.title('Age Distribution (Categories)')
    plt.tight_layout()
    plt.savefig('figures/age_disease.png')
    plt.clf()

    # Distribution of ages for different class with independent normalization.
    sns.violinplot(data=df, x='Age', y='Disease category', orient='h', palette='crest')
    plt.yticks(ticks=range(4, -1, -1), labels=disease_label)
    plt.ylabel('Category')
    plt.title('Age Distribution (Categories)')
    plt.tight_layout()
    plt.savefig('figures/age_disease_noralized.png')
    plt.clf()


def pitch_distribution(path: str) -> None:
    df = pd.read_csv(path)
    disease_label = ['Phonotrauma', 'Incomplete Glottic Closure', 'Vocal Palsy', 'Neoplasm', 'Normal']
    disease_label.reverse()

    peak = []
    for ID in df.ID:
        file = os.path.join(os.path.dirname(path), 'raw', ID+'.wav')
        mel = average_mel_spectrogram(file)[:20]
        peak.append(np.argmax(mel))
    df['melpeak'] = peak
    healthy = df[df['Disease category'] == 5].copy()

    # Pitch of Male and Female.
    plt.figure(figsize=(16, 9))
    sns.violinplot(healthy, x='melpeak', y='Sex', orient='h')
    plt.title('Peak Mel-Frequency Differences')
    plt.xlabel('Peak Mel-Frequency')
    plt.tight_layout()
    plt.savefig('figures/pitch_sex_healthy.png')
    plt.clf()

    # Pitch of different diseases (male only).
    sns.violinplot(df[df.Sex == 1], x='melpeak', y='Disease category', orient='h', palette='crest')
    plt.yticks(ticks=range(4, -1, -1), labels=disease_label)
    plt.title('Peak Mel-Frequency Differences (Male)')
    plt.xlabel('Peak Mel-Frequency')
    plt.tight_layout()
    plt.savefig('figures/pitch_disease_male.png')
    plt.clf()

    # Pitch of healthy people along ages. TODO: Too few healthy data.
    df['Age Categories'] = pd.cut(df.Age, np.arange(20, 80, 10))
    sns.swarmplot(df[df.Sex == 1], x='Age Categories', y='melpeak')
    plt.savefig('figures/pitch_age_healthy.png')
    plt.clf()
    return


def average_mel_spectrogram(path: str):
    fs = librosa.get_samplerate(path)
    y, fs = librosa.load(path, sr=fs)
    y = y / np.max(np.abs(y))
    mel = librosa.feature.melspectrogram(y=y, sr=fs, fmax=8000)
    return mel.mean(axis=1)


def check_data_correctness(path: str, n_sample: np.ndarray) -> int:
    df = pd.read_csv(path)

    # Number of sample for each class.
    for i in range(len(n_sample)):
        if df.query(f'`Disease category` == {i + 1}').shape[0] != n_sample[i]:
            print(f'class {i + 1} with incorrect number of samples')

    # Correctness of filenames.
    for ID in df.ID:
        filename = os.path.join(os.path.dirname(path), 'raw', ID+'.wav')
        if not os.path.isfile(filename):
            print(f'Cannot find {ID}.wav')
    return 0


if __name__ == '__main__':
    # n_sample = [536, 220, 168, 44, 32]
    # check_data_correctness('Data/Train/data_list.csv', n_sample)

    age_distributions('Data/Train/data_list.csv')

    # This will take a long time to compute.
    pitch_distribution('Data/Train/data_list.csv')
