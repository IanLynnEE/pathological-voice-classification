# <center>Pathological Voice Classification

<div align="center">
    <img src="./figures/logo.png" alt="Project Logo" width="200">
</div>

## Introduction

In recent years, increasing work pressures and fast-paced lifestyles have brought attention to modern diseases, including voice disorders that commonly affect professionals such as teachers, salespeople, and lecturers. Traditional detection methods are invasive and require specialized equipment, making early diagnosis challenging and often delayed. The COVID-19 pandemic has further highlighted the need for non-contact approaches due to the risk of droplet transmission during endoscopic examinations.

This project aims to develop a non-contact voice disorder detection model using feature engineering and machine learning. By analyzing voice signals alongside medical history records, our model can effectively detect and classify laryngeal conditions, enabling early intervention. This innovation offers significant benefits, especially for those who may delay seeking medical attention due to time constraints or safety concerns. The goal is to advance medical diagnostics in vocal health, using AI to reduce misdiagnosis rates and provide timely, accessible care to those in need.

This project achieved **remarkable recognition** in the [_2023 AI CUP - Multimodal Pathological Voice Classification Competition_](https://tbrain.trendmicro.com.tw/Competitions/Details/27?fbclid=IwAR31bemJgWn79XYhA8zLk8ThJSnN2laTMwrIWI4aL6zQD6Y1HW2eCrLoXhk), securing a top position on the leaderboard by ranking **second out of 371 teams**.

## Project Contributors

- [Chun Lin, Huang](https://www.facebook.com/profile.php?id=100002217587773)
- [Li Cheng, Chien](https://www.linkedin.com/in/li-cheng-chien/)
- [Tsung Wei, Lin](https://www.linkedin.com/in/tsung-wei-lin-862250234/)

## Requirements

Python >= 3.10.

```shell
pip install -r requirements.txt
```

## Reproduction

The best results on the private test set were achieved by two random forest classifiers with the vocal-tract-area (VTA) estimator [1]. Tuned hyperparameters are listed in `config.py`. Please note that the code uses `eval()` and `pickle` for convenience. **Users are solely responsible for verifying the correctness and safety of all code and arguments.**

```shell
python3 train_rf.py \
    --csv_path ${path_to_the_training_data_list} \
    --audio_dir ${directory_that_stores_training_audios} \
    --test_csv_path ${path_to_the_test_data_list} \
    --test_audio_dir ${directory_that_stores_test_audios} \
    --output ${path_of_the_output_file}
```

We recommend storing the data in the following manner:

```shell script
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

If data were stored in the recommended manner, a shortcut is provided to train a model and test on the private dataset:

```shell
python3 train_rf.py --prefix Private --output ${path_of_the_output_file}
```

Please note that the difference in random states of the random forest classifiers may yield very different results.

<!--
We constructed various models to classify pathological voice data, exploring different model architectures to maximize performance. Our experiments included:

- **Tree booster-based models**: Leveraging gradient boosting methods, including XGBoost and LightGBM, to handle structured data effectively.
- **CNN-based models**: Utilizing convolutional neural networks to capture spatial patterns and local features in spectrogram representations of voice signals.
- **RNN-based models**: Applying recurrent neural networks, including GRUs and LSTMs, to model sequential dependencies in voice signal data.
- **Transformer-based models**: Using transformer encoders to capture long-range dependencies and complex relationships in the data.

Among these, the model that achieved the best performance in the competition is highlighted below.
 -->

## Methodology

### Preprocessing

Since many well-developed models require fixed-length input, audio samples were sliced into fixed-length frames without overlapping. This operation helpfully increased training samples, yet the assumption that sliced frames hold the same label (properties) as the original sample was made in the process. Details can be found in `preprocess.py`.

### Audio Features Extraction -- Vocal Tract Area (VTA) Calculation

#### Motivation

For the pathological classification task, traditional feature extraction methods such as MFCC might not be suitable. MFCC tries to mimic the behavior of human ears, which should be a good strategy for tasks that human beings are good at. However, applying the cepstrum eliminates the influence of reflection in the vocal tract, which may be helpful information when classifying pathological voices. Thus, another feature extraction method was implemented: the vocal-tract-area (VTA) estimator [1].

#### Idea of VTA

VTA relies on the idea of compensation. For patients with voice disorders, it was assumed that they would need to use organs other than the vocal folds to compensate for the difficulty of making sounds. The reflection pattern, thus, would be different from time to time. By analyzing the differences in those patterns, the illness features for identifying the voice disorders would be extracted.

#### Details about VTA

Let the sound made at time $t$ be $x_t$. We assumed that it can be reconstructed by a linear combination of $M$ data points before time $t$.

$$
\hat x_t = \sum_{i=1}^{M}a_ix_{t-i}
$$

The mean square error can be found as the following:

$$E=\sum_{t=M+1}^{N}(x_t-\sum_{i=1}^{M}a_ix_{t-i})^2$$

Make the gradient of $a_k$ to be $0$ in order to the get the minimum error:

$$
\frac{\partial E}{\partial a_k}=0\Rightarrow \sum_{t=M+1}^{N}x_{t-k}\sum_{i=1}^{M}a_ix_{t-i}=\sum_{t=M+1}^{N}x_tx_{t-k}
$$

Let $r(k)=\sum_{j=0}^{N-k-1}x_nx_{n+k}$, and we assumed $r(.)$ is $0$ with every negative index. When $N$ is large enough, we can have an approximate formula in a matrix form.

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

The dataset used in this project shows a substantial imbalance issue. To resolve this, **Balanced Random Forest classifiers** were used from the `imbalanced-learn` package.

However, observations of overfitting or underfitting were still made when a single modality (audio or clinical features) was adopted. Therefore, **Late fusion** was introduced to reduce the problem. The results of the validation set were better than those of early fusion. The architecture of training and testing late fusion RFs can be found in `train_rf.py`.

### Voting

For the audio classifier, a prediction will be made for each frame. Hence, there might be multiple predictions for one audio sample. To summarize the predictions, the predicted probabilities of a sample were determined by the average of the predicted probabilities of corresponding frames.

The design philosophy of taking average was to compensate for the assumption made in the preprocessing. A sample can still be classified into a class if one frame yields a confident prediction while results from others disagree with little confidence.

Another advantage of averaging is its simplicity. The exact mechanism was used to summarize the predictions from the audio and clinical classifiers with very little modification.

More voting options can be found in `utils.py`, and we do believe better results can be achieved by modifying the voting mechanism, providing that models have reasonable accuracy.

## Observations and Future Works

Other models were implemented in different files/branches, including **early fusion Balanced Random Forest**, **Feedforward Neural Network (FNN)**, **Convolution Neural Network (CNN)**, **Recurrent Neural Network (RNN)**, **Transformer**, etc. The performance was not much of an improvement compared to the late fusion Balanced Random Forest, and we found the effect of random state in both train-test split and model initialization were more pronounced than the difference in models. This might be due to the small size of the dataset for multiclass classification and the poor performance of the models.

Considering the overall performance had room for improvement, we are optimistic that **RNN-based models** or **Transformer** can significantly enhance our results. Thus, we look forward to constructing an RNN-based model (LSTM, GRU) with attention layers or transformers to tackle this task.

## References

[1] G. Muhammad *et al.*, ‘Automatic voice pathology detection and classification using vocal tract area irregularity’, *Biocybernetics and Biomedical Engineering*, vol. 36, no. 2, pp. 309–317, 2016.

[2] Logo inspired by [Free Icon](https://freeiconsite.com) from Freepik and Reddie.

<!-- <a href="https://www.flaticon.com/free-icons/machine-learning" title="machine learning icons">Machine learning icons created by Reddie - Flaticon</a> -->
<!-- <a href="https://www.flaticon.com/free-icons/voice" title="voice icons">Voice icons created by Freepik - Flaticon</a> -->
<!-- <a href="https://www.flaticon.com/free-icons/stethoscope" title="stethoscope icons">Stethoscope icons created by Freepik - Flaticon</a> -->
