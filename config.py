import argparse

def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--csv_path', type=str, default='Data/Train/data_list.csv')
    parser.add_argument('--audio_dir', type=str, default='Data/Train/raw')
    parser.add_argument('--feature_extraction', type=str, default='vta', choices=['mfcc', 'vta', 'empty'])
    parser.add_argument('--test_csv_path', type=str, default='None')
    parser.add_argument('--test_audio_dir', type=str, default='None')
    parser.add_argument('--temporal', action='store_true')
    parser.add_argument('--do_smote', action='store_true')
    parser.add_argument('--smote_strategy', type=str, default='SMOTE')

    parser.add_argument('--fs', type=int, default=22050)
    parser.add_argument('--frame_length', type=int, default=3675)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mfcc', type=int, default=13)

    parser.add_argument('--n_tube', type=int, default=21)
    parser.add_argument('--vta_window_length', type=int, default=175)

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')

    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--batch_size', type=int, default=256)

    args = parser.parse_args()
    args.max_features = int(args.max_features) if args.max_features.isdecimal() else args.max_features
    args.test_audio_dir = args.test_audio_dir if args.test_audio_dir != 'None' else args.audio_dir
    return args