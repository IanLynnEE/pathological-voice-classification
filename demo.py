import argparse
import os
import platform     # Check macOS or not.
import shutil       # Check if terminal-notifier is installed.
import time

import librosa
import numpy as np
import torch

from train_nn import get_dataloader
from inference import load_model_and_get_result
from preprocess import get_audio_features, get_1d_data
from utils import summary


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary_task', action='store_true', default=True)
    parser.add_argument('--model', type=str, default='AudioNN_99')
    parser.add_argument('--feature_extraction', type=str, default='vta')
    parser.add_argument('--fs', type=int, default=24000)
    parser.add_argument('--frame_length', type=int, default=4000)
    parser.add_argument('--n_tube', type=int, default=21)
    parser.add_argument('--vta_window_length', type=int, default=160)
    return parser.parse_args()


def main():
    args = get_config()
    record_dir = '../Movies/Omi Screen Recorder'
    time_limit = 900
    device = torch.device('cpu')

    if not args.binary_task:
        raise NotImplementedError('Not implemented for non-binary task.')

    # Continually check if there is file in the target directory.
    for i in range(time_limit):
        print(f'{time_limit-i:4d}s till session end.', end='\r')
        # If there is a file, read it and do the prediction.
        if len(os.listdir(record_dir)) > 0:
            filename = os.path.join(record_dir, os.listdir(record_dir)[-1])

            if filename.endswith('.DS_Store'):
                os.remove(filename)
                continue

            raw, c_fake, y_fake, id_fake = read_a_file(filename, args.frame_length, args.fs)
            try:
                x = get_audio_features(raw, args)

                # TODO: Do any preprocessing here.
                _, _, _, _, _, x = get_1d_data(x)

                dataloader = get_dataloader(x, c_fake, y_fake, batch_size=1, binary=args.binary_task)
                y_prob = load_model_and_get_result(args.model, x, c_fake, dataloader, device=device, binary_task=True)
                pred = summary(y_fake, y_prob, id_fake)
                if pred.pred[0] == 0:
                    notify('Prediction', 'We did not detect any sign of voice disorder.')
                else:
                    notify('Prediction', 'We advice you to seek medical help.')
            except Exception as e:
                notify('Recording Error', e)
            finally:
                os.remove(filename)
        time.sleep(1)
    return


def read_a_file(filename: str, frame_length: int, fs: int = 48000, clinical_dim: int = 5, binary: bool = True):
    audio, _ = librosa.load(filename, sr=fs, mono=False)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    n_channels = audio.shape[1] if audio.ndim == 2 else 1

    n_frames = int(audio.shape[0] / frame_length) * n_channels
    x = np.zeros((n_frames, frame_length))

    print(f'Number of frames: {n_frames}, number of channels: {n_channels}.')

    # Audio
    for j in range(n_frames):
        x[j] = audio[j * frame_length:(j + 1) * frame_length]

    # Fake ids, clinical data, and y.
    ids = np.zeros(n_frames, dtype=np.int_)
    y = np.ones(n_frames, dtype=np.int_)
    c = np.zeros((n_frames, clinical_dim))
    return x, c, y, ids


def notify(title, text, url=None, sound='default'):
    if shutil.which('terminal-notifier') is None:
        os.system('''
            osascript -e 'display notification "{}" with title "{}" sound name "{}"'
        '''.format(text, title, sound))
    elif url is None:
        os.system('''
            terminal-notifier -title '{}' -message '{}' -sound '{}'
        '''.format(title, text, sound))
    else:
        os.system('''
            terminal-notifier -title '{}' -message '{}' -open '{}' -sound '{}'
        '''.format(title, text, url, sound))
    return


if __name__ == '__main__':
    if platform.system() != 'Darwin':
        raise NotImplementedError('Not implemented for non-macOS.')
    try:
        main()
    except KeyboardInterrupt:
        print('\r', 'Ctrl-C pressed. Exiting...')
