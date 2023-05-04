import os
import sys

import unittest
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

sys.path.insert(1, os.path.abspath('.'))
print(sys.path[1])
from preprocess import read_audio_files

class TestReadAudioFiles(unittest.TestCase):

    def setUp(self):
        # Create test audio files
        self.fs = 22050
        self.win_length = 1024
        self.sound_dir = 'test_data/raw'
        if not os.path.exists(self.sound_dir):
            os.makedirs(self.sound_dir)
        self.audio_length = 5  # seconds
        for i in range(3):
            audio = np.random.randn(self.fs * self.audio_length)
            # librosa.output.write_wav(os.path.join(self.sound_dir, f'test_audio_{i}.wav'), audio, sr=self.fs)
            sf.write(os.path.join(self.sound_dir, f'test_audio_{i}.wav'), audio, samplerate=self.fs)

        # Create test CSV file
        test_data = {'ID': [f'test_audio_{i}' for i in range(3)],
                     'Disease category': [1, 2, 3]}
        self.df = pd.DataFrame(data=test_data)
        self.csv_path = 'test_data/test_csv.csv'
        self.df.to_csv(self.csv_path, index=False)

    def test_output_shapes(self):
        # Call function under test
        x, y = read_audio_files(csv_path=self.csv_path, fs=self.fs, win_length=self.win_length)

        # Assert outputs are correct shape
        self.assertEqual(x.shape[1], 1024)
        self.assertEqual(y.shape[0], x.shape[0])

    def test_output_contents(self):
        # Call function under test
        x, y = read_audio_files(csv_path=self.csv_path, fs=self.fs, win_length=self.win_length)

        # Assert contents of x and y are valid
        self.assertTrue(np.all(x >= -1) and np.all(x <= 1))  # audio should be normalized between -1 and 1
        self.assertTrue(np.all(np.isin(y, [1, 2, 3])))  # labels should match test CSV file

    def tearDown(self):
        # Remove test audio files and CSV file
        for i in range(3):
            os.remove(os.path.join(self.sound_dir, f'test_audio_{i}.wav'))
        os.remove(self.csv_path)

if __name__ == "__main__":
    unittest.main()