# Pathological Voice Classification

## Requirements

```shell
python >= 3.10

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

*Please note that the difference in random states of the random forest classifiers may yield very different results.*

## Model Design

### Preprocessing

Since lots of well-developed models require fix length input, audio samples were sliced into fixed length frames without overlapping. This operation helpfully increased the amount of training data, yet the assumption that sliced frames hold the same label (properties) as the original sample was made in the process. Details can be found in `preprocess.py`.

### Audio Features Extraction -- Vocal Tract Area (VTA) Calculation

#### Motivation

To be frank, Pathological classification is quitely different from traditional voice classification. We consider that the main assumption of traditional MFCC is not suitable for the task. MFCC tries to mimic what human ear hears, which eliminates the influence of reflection in the vocal tract. This may abandon lots of information when it comes to our case. We then look for a better way of feature extraction -- ***VTA***, which was published in Biocybernetics and Biomedical Engineering in 2016.

#### Idea of VTA

The idea of VTA is simple. For a patient with any voice pathology, he/she will need to use muscle other from throat for compensation when making a sound. It means that the reflection pattern in the throat would be different from time to time. From analyzing the differnces of those pattern, we can get the illness features to identify the pathology.

#### Mathematical detail about VTA

To capture the pattern of reflected voice on a time series. The paper assumed that the sound we make at time t is $x_t$ which can be written as a linear combination of $M$ data point before it.

$$x_t=\sum_{i=1}^{M}a_ix_{t-i}$$

Take the square error and summation all the point we get the loss as the following:

$$E=\sum_{t=M+1}^{N}(x_t-\sum_{i=1}^{M}a_ix_{t-i})^2$$

We make the gradient of $a_k$ to be $0$ in order the get the minimum error:

$$\frac{\partial E}{\partial a_k}=0\Rightarrow \sum_{t=M+1}^{N}x_{t-k}\sum_{i=1}^{M}a_ix_{t-i}=\sum_{t=M+1}^{N}x_tx_{t-k}$$

Set $r(k)=\sum_{j=0}^{N-k-1}x_nx_{n+k}$ ,and assume $r$ as $0$ with negative index. When $N$ is large enough, we can have a approximate formula write in a matrix form.

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

Then we can get the coefficient $a_i$ by multipling the RHS with invesre of the square matrix.

### Random Forest Classifiers

According to the imbalanced issue in the given dataset, traditional Random Forest cannot class appropriately. Therefore, we use Balanced Random Forest to eliminate the influence.

Another issue is that it makes overfitting by RF with only clinical features and underfitting by that with only audio features. To figure out this problem, we use late fusion, which combines the effect of the outputs from the two models. The results verified that it does eliminate the problems of each other. The architecture of RF with different data can find in `train_rf.py`, while the late fusion method can find in `utils.py`

### Voting

For the audio classifier, a prediction will be made for each frame. Hence, there might be multiple predictions for one audio sample. To summarize the predictions, the predicted probabilities of a sample were determined by the average of the predicted probabilities of corresponding frames.

The design philosophy of taking average was to compensate for the assumption made in the preprocessing. A sample can still be classified into a class if one frame yields a confident prediction, while results from other frames disagreed with little confidence.

Another advantage of the average is simplicity. The same mechanism was used in summarizing the predictions from the audio classifier and the clinical classifier with very little modification.

More voting options can be found in `utils.py`, and we do believe better results can be achieved by modifying the voting mechanism, providing that models have reasonable accuracy.

## Observations and Future Works

We compared different models, including Random Forest (RF), Feedforward Neural Network (FFN), Convolution Neural Network (CNN), and Recurrent Neural Network (RNN). The result shows that RF still gets the best performance, while FFN can keep pace with it but is easy to overfit. On the other hand, CNN and RNN are relatively worse.

We consider that the audio features can decode by an RNN-based model. Besides, we think that adding Transformer can improve the performance based on the concept of feature extraction. Thus, we look forward to constructing an RNN-based model (LSTM, GRU) with an attention layer or Transformer applied to this issue.

## References

[1] G. Muhammad *et al.*, ‘Automatic voice pathology detection and classification using vocal tract area irregularity’, *Biocybernetics and Biomedical Engineering*, vol. 36, no. 2, pp. 309–317, 2016.
