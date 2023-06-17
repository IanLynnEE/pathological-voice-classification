import argparse
import os


def get_config() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--torch_seed', type=int, default=2)
    parser.add_argument('--rf_seed', type=int, default=21)
    parser.add_argument('--csv_path', type=str, default='Data/Train/data_list.csv')
    parser.add_argument('--audio_dir', type=str, default='Data/Train/raw')
    parser.add_argument('--valid_csv_path', type=str, default='Data/Public/data_list.csv')
    parser.add_argument('--valid_audio_dir', type=str, default='Data/Public/raw')
    parser.add_argument('--binary_task', action='store_true', default=False)
    parser.add_argument('--output', type=str, default=None)

    parser.add_argument('--model', type=str, default='ClinicalNN', nargs='*')
    parser.add_argument('--feature_extraction', type=str, default='vta',
                        choices=['mfcc', 'vta', 'clinical_only', 'raw'])

    parser.add_argument('--fs', type=int, default=22050)
    parser.add_argument('--frame_length', type=int, default=3675)
    parser.add_argument('--n_fft', type=int, default=2048)
    parser.add_argument('--n_mfcc', type=int, default=13)

    parser.add_argument('--n_tube', type=int, default=21)
    parser.add_argument('--vta_window_length', type=int, default=175)
    parser.add_argument('--n_frames', type=int, default=50,
                        help='number of frames for each sample, only enable when frame_length is 0')

    # Following two are invalid, as they were proved to be not helpful.
    parser.add_argument('--do_smote', action='store_true')
    parser.add_argument('--smote_strategy', type=str, default='SMOTE')

    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--max_depth', type=int, default=None)
    parser.add_argument('--min_samples_split', type=int, default=2)
    parser.add_argument('--min_samples_leaf', type=int, default=1)
    parser.add_argument('--max_features', type=str, default='sqrt')

    # Following default values were designed for ClinicalNN.
    # AudioCNN usually needs much smaller learning rate, e.g. 8e-6.
    parser.add_argument('--best_score', type=float, default=0.6, help='threshold for saving best model')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=12e-5)
    parser.add_argument('--pct_start', type=float, default=0.3)
    parser.add_argument('--div_factor', type=int, default=4)
    parser.add_argument('--final_div_factor', type=int, default=10000)
    parser.add_argument('--three_phase', action='store_true', default=False)

    args = parser.parse_args()

    # Max features for Random Forest, default is `sqrt`.
    args.max_features = int(args.max_features) if args.max_features.isdecimal() else args.max_features

    # Handle model name. If only one model is provided, it will be a string. Otherwise, it will be a list.
    args.model = args.model[0] if len(args.model) == 1 else args.model
    if len(args.model) == 0:
        raise argparse.ArgumentError(None, 'Please provide at least a model name')

    # Set default output.
    if args.output is None:
        prefix = os.path.basename(os.path.dirname(args.valid_csv_path))
        model = f'{args.model[0]}_{args.model[1]}' if isinstance(args.model, list) else args.model
        if args.feature_extraction in model:
            args.output = f'{prefix}_{model}'
        else:
            args.output = f'{prefix}_{model}_{args.feature_extraction}'
    return args
