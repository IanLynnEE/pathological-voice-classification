import argparse


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--torch_seed', type=int, default=2)
    parser.add_argument('--rf_seed', type=int, default=21)
    parser.add_argument('--csv_path', type=str, default='Data/Train/data_list.csv')
    parser.add_argument('--audio_dir', type=str, default='Data/Train/raw')
    parser.add_argument('--feature_extraction', type=str, default='vta', choices=['mfcc', 'vta', 'clinical_only'])
    parser.add_argument('--prefix', type=str, default='None', choices=['Public', 'Private'])
    parser.add_argument('--test_csv_path', type=str, default='None')
    parser.add_argument('--test_audio_dir', type=str, default='None')

    parser.add_argument('--fs', type=int, default=22050)
    parser.add_argument('--frame_length', type=int, default=3675)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mfcc', type=int, default=13)

    parser.add_argument('--n_tube', type=int, default=21)
    parser.add_argument('--vta_window_length', type=int, default=175)

    parser.add_argument('--do_smote', action='store_true')
    parser.add_argument('--smote_strategy', type=str, default='SMOTE')

    parser.add_argument('--single_rf', action='store_true', default=False)
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')

    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--div_factor', type=int, default=4)
    parser.add_argument('--final_div_factor', type=int, default=10000)
    parser.add_argument('--three_phase', action='store_true', default=False)

    args = parser.parse_args()
    args.max_features = int(args.max_features) if args.max_features.isdecimal() else args.max_features
    if args.prefix == 'Public':
        args.test_csv_path = 'Data/Public/data_list.csv'
        args.test_audio_dir = 'Data/Public/raw'
    elif args.prefix == 'Private':
        args.test_csv_path = 'Data/Private/data_list.csv'
        args.test_audio_dir = 'Data/Private/raw'
    else:
        pass
    args.test_audio_dir = args.test_audio_dir if args.test_audio_dir != 'None' else args.audio_dir
    return args
