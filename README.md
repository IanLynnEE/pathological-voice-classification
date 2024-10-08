# Pathological Voice Classification

## Requirements

Python >= 3.10.

```shell
pip install -r requirements.txt
```

## Reproduction

The best score on the private test set was achieved by two random forest classifiers with the vocal-tract-area (VTA) estimator [1]. Default values were used except for parameters listed in `config.py`. For convenience, `eval()` and `pickle` were used in the code. **Users are solely responsible for verifying the correctness and safety of the code and arguments.**

To reproduce the results, one can run the following commands:

```shell
python3 train_rf.py \
    --csv_path ${path_to_the_training_data_list} \
    --audio_dir ${directory_that_stores_training_audios} \
    --test_csv_path ${path_to_the_test_data_list} \
    --test_audio_dir ${directory_that_stores_test_audios} \
    --output ${path_of_the_output_file}
```

No external training nor validation dataset was used.

Shortcuts are available if data were stored in the following manner:

```shell
.
└── Data/
    ├── Train/
    │   ├── data_list.csv
    │   └── raw/
    │       ├── audio.wav
    │       └── ...
    ├── Public/
    │   ├── data_list.csv
    │   └── raw/
    │       ├── audio.wav
    │       └── ...
    └── Private/
        ├── data_list.csv
        └── raw/
            ├── audio.wav
            └── ...
```

The shortcut for the private test is:

```shell
python3 train_rf.py --prefix Private --output ${path_of_the_output_file}
```

Please note that the difference in random states of the random forest classifiers may yield very different results.

## Model Design

### Preprocessing

Since lots of well-developed models require fix length input, audio samples were sliced into fixed length frames without overlapping. This operation helpfully increased the amount of training data, yet the assumption that sliced frames hold the same label (properties) as the original sample was made in the process. Details can be found in `preprocess.py`.

### Audio Features Extraction -- Vocal Tract Area (VTA) Calculation

#### Motivation

For the pathological classification task, traditional feature extraction methods such as MFCC might not be suitable. MFCC tries to mimic the behavior of human ears, which should be a good strategy for tasks that human beings are good at. However, the application of the cepstrum eliminates the influence of reflection in the vocal tract. This may abandon useful information when it comes to classifying pathological voices. Thus, another feature extraction method was implemented: the vocal-tract-area (VTA) estimator [1].

#### Idea of VTA

VTA relays on the idea of compensation. For patients with voice disorders, an assumption was made that they would need to use organs other than the vocal folds to compensate for the difficulty of making sounds. The reflection pattern, thus, would be different from time to time. By analyzing the differences in those patterns, the illness features for identifying the voice disorders would be extracted.

#### Details about VTA

Let the sound made at time $t$ be $x_t$. We assumed that it can be reconstructed by a linear combination of $M$ data points before time $t$.

$$
\hat x_t = \sum_{i=1}^{M}a_ix_{t-i}
$$

The mean square error can be found as the following:

$$E=\sum_{t=M+1}^{N}(x_t-\sum_{i=1}^{M}a_ix_{t-i})^2$$

Make the gradient of $a_k$ to be $0$ in order the get the minimum error:

$$
\frac{\partial E}{\partial a_k}=0\Rightarrow \sum_{t=M+1}^{N}x_{t-k}\sum_{i=1}^{M}a_ix_{t-i}=\sum_{t=M+1}^{N}x_tx_{t-k}
$$

Let $r(k)=\sum_{j=0}^{N-k-1}x_nx_{n+k}$, and we assumed $r(.)$ is $0$ with every negative index. When $N$ is large enough, we can have an approximate formula written in a matrix form.

$$
\begin{bmatrix}
r(0) & r(1) & ... & r(M-2) & r(M-1)\\
r(1) & r(0) & ... & r(M-3) & r(M-2)\\
... & ... & ... & ... & ...\\
r(M-2) & r(M-3) & ... & r(0) & r(1)\\
r(M-1) & r(M-2) & ... & r(1) & r(0)\\
\end{bmatrix}
\begin{bmatrix}
a_1\\
a_2\\
...\\
a_{M-1}\\
a_{M}\\
\end{bmatrix}=
\begin{bmatrix}
r(1)\\
r(2)\\
...\\
r(M-1)\\
r(M)\\
\end{bmatrix}
$$

Then we can get the coefficient $a_i$ by multiplying the RHS with the inverse of the square matrix.

### Random Forest Classifiers

Considering the imbalanced issue in the given dataset, the Balanced Random Forest classifiers, provided by `imbalanced-learn`, were used.

The problem of overfitting by using only clinical features and underfitting by using only audio features cannot be solved by simply adopting the Balanced Random Forest classifiers. Late fusion was introduced in the hope to reduce the problem. The results of the validation set were better than early fusion. The architecture of training and testing late fusion RFs can find in `train_rf.py`.

### Voting

For the audio classifier, a prediction will be made for each frame. Hence, there might be multiple predictions for one audio sample. To summarize the predictions, the predicted probabilities of a sample were determined by the average of the predicted probabilities of corresponding frames.

The design philosophy of taking average was to compensate for the assumption made in the preprocessing. A sample can still be classified into a class if one frame yields a confident prediction, while results from other frames disagreed with little confidence.

Another advantage of averaging is its simplicity. The same mechanism was used in summarizing the predictions from the audio classifier and the clinical classifier with very little modification.

More voting options can be found in `utils.py`, and we do believe better results can be achieved by modifying the voting mechanism, providing that models have reasonable accuracy.

## Observations and Future Works

Other models were implemented in other files/branches, including early fusion Balanced Random Forest, Feedforward Neural Network (FNN), Convolution Neural Network (CNN), Recurrent Neural Network (RNN), etc. The performance was not much of an improvement compared to the late fusion Balanced Random Forest, and we found the effect of random state in both train-test split and model initialization were more pronounced than the difference in models. This might be due to the small size of the dataset for multiclass classification and the poor performance of the models.

Considering the overall performance was poor, we still believe that RNN-based models or the Transformer can still improve the performance. Thus, we look forward to constructing an RNN-based model (LSTM, GRU) with attention layers or transformers to tackle this task.

## References

[1] G. Muhammad *et al.*, ‘Automatic voice pathology detection and classification using vocal tract area irregularity’, *Biocybernetics and Biomedical Engineering*, vol. 36, no. 2, pp. 309–317, 2016.
