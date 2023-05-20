# Pathological Voice Classification

## Requirements

TODO: specify that we need Python > 3.10.

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
    │       │── audio.wav
    │       └── ...
    ├── Public/
    │   ├── data_list.csv
    │   └── raw/
    │       │── audio.wav
    │       └── ...
    └── Private/
        ├── data_list.csv
        └── raw/
            │── audio.wav
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

### Audio Features Extraction

TODO

### Random Forest Classifiers

TODO: To address the imbalance issue,...

### Voting

For the audio classifier, a prediction will be made for each frame. Hence, there might be multiple predictions for one audio sample. To summarize the predictions, the predicted probabilities of a sample were determined by the average of the predicted probabilities of corresponding frames.

The design philosophy of taking average was to compensate for the assumption made in the preprocessing. A sample can still be classified into a class if one frame yields a confident prediction, while results from other frames disagreed with little confidence.

Another advantage of the average is simplicity. The same mechanism was used in summarizing the predictions from the audio classifier and the clinical classifier with very little modification.

More voting options can be found in `utils.py`, and we do believe better results can be achieved by modifying the voting mechanism, providing that models have reasonable accuracy.

## Observations and Future Works

Optional

## References

[1] G. Muhammad *et al.*, ‘Automatic voice pathology detection and classification using vocal tract area irregularity’, *Biocybernetics and Biomedical Engineering*, vol. 36, no. 2, pp. 309–317, 2016.
