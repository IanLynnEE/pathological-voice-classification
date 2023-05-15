## Dataset Path
```
─── Data/                
    ├── Train/
    │   ├── data_list.csv
    │   └── raw/
    │       │── audio.wav
    │       └── ...
    └── Public/
        ├── data_list.csv
        └── raw/
            │── audio.wav
            └── ...
```

## Install Packages
```
pip install -r requirements.txt
```

## Use SMOTE
```
python mainfile.py --do_smote \
                   --smote_strategy SMOTE
```