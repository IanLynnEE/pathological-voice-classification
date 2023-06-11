from collections import Counter
import sys

import numpy as np
import pandas as pd
import torch


def get_SMOTE(X, y, seed, SMOTE_strategy, categorical_features=None) -> tuple[np.ndarray, np.ndarray]:
    print(f'Original Dataset shape {Counter(y)}')
    print(f'Over sampling with {SMOTE_strategy.__name__}')
    if SMOTE_strategy.__name__ == 'SMOTENC':
        sm = SMOTE_strategy(random_state=seed, categorical_features=categorical_features)
    else:
        sm = SMOTE_strategy(random_state=seed)
    sm_X, sm_y = sm.fit_resample(X, y)
    print(f'Resampled Dataset shape {Counter(sm_y)}')
    return (sm_X, sm_y)


def summary(y_truth, y_prob, ids, tricky_vote=False, to_left=False):
    if isinstance(y_prob, tuple):
        prob_sum = None
        idx = [0, 1] if y_prob[0].shape[1] == 2 else [1, 2, 3, 4, 5]
        for prob in y_prob:
            a = np.c_[ids, prob]
            a = pd.DataFrame(a, columns=['ID', *idx])
            if prob_sum is None:
                prob_sum = a.groupby('ID').agg(pd.Series.mean)
            else:
                prob_sum += a.groupby('ID').agg(pd.Series.mean)
        results = prob_sum.idxmax(axis='columns').to_frame(name='pred')
        if tricky_vote or to_left:
            raise NotImplementedError('y_prob is a tuple.')
    else:
        y_pred = np.argmax(y_prob, axis=1) + np.sign(y_prob.shape[1] - 2)
        results = pd.DataFrame({'ID': ids, 'pred': y_pred})
        if tricky_vote and not to_left:
            results = results.groupby('ID').pred.agg(max).to_frame()
        elif tricky_vote and to_left:
            results = results.groupby('ID').pred.agg(min).to_frame()
        elif to_left:
            results = results.groupby('ID').pred.agg(lambda x: min(pd.Series.mode(x))).to_frame()
        else:
            results = results.groupby('ID').pred.agg(lambda x: max(pd.Series.mode(x))).to_frame()
    ground_truth = pd.DataFrame({'ID': ids, 'truth': y_truth})
    ground_truth = ground_truth.groupby('ID').truth.agg(pd.Series.mode).to_frame()
    return results.merge(ground_truth, how='inner', on='ID', validate='1:1')


def save_checkpoint(epoch, model, optimizer, scheduler=None):
    path = f'runs/{model.__class__.__name__}_{epoch}.pt'
    if scheduler is None:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        return
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)
    return path


def merge_and_check():
    template_path = sys.argv[3]
    template = pd.read_csv(template_path, header=None, names=['ID', 'fake'])
    f1 = pd.read_csv(sys.argv[1], header=None, names=['ID', 'pred'])
    f2 = pd.read_csv(sys.argv[2], header=None, names=['ID', 'pred'])
    df = pd.concat([f1, f2], ignore_index=True, verify_integrity=True)
    print('Duplicate IDs:', df.duplicated(subset=['ID']).any())
    if df.ID.isin(template.ID).all() and template.ID.isin(df.ID).all():
        print('Union Checked.\nNumber of Samples:', df.shape[0])
        df.set_index('ID').to_csv(sys.argv[4], header=False)
        return
    print('Mismatched IDs!')


if __name__ == '__main__':
    merge_and_check()
