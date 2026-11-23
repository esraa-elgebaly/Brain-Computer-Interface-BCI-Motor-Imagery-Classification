# BCI motor imagery classification (LEFT vs RIGHT)

An end-to-end pipeline that tries to tell, from 4-channel EEG alone, whether a person is **imagining** a left-hand or a right-hand movement. It covers loading, preprocessing, EDA, three models, evaluation and a Streamlit demo.

Everything lives in one notebook, [`BCI_MI_Pipeline.ipynb`](BCI_MI_Pipeline.ipynb), which follows the task document step by step. The notebook and the demo app share one preprocessing module, so a trial is processed the same way in training and in the app.

## Result in one paragraph

The pipeline runs end to end, but **every model scores at chance level** (AUC ≈ 0.50). The models are not the cause. `labels.csv` holds 2400 labels with no file names for 2160 trial files. We found 36 pairs of byte-identical recordings, and only 12 of those pairs carry the same label. The EEG also shows no LEFT/RIGHT difference in the mu/beta bands on C3/C4. The labels are not linked to the right files, so there is nothing for a model to learn. Stage 3.4 (data quality) and Stage 6.3 (tests against chance) of the notebook give the full evidence.

| Model | AUC | Balanced accuracy | Accuracy |
|---|---|---|---|
| CSP + SVM (baseline) | 0.504 ± 0.029 | 0.508 ± 0.021 | 0.509 ± 0.030 |
| EEGNet | 0.502 ± 0.019 | 0.496 ± 0.012 | 0.495 ± 0.012 |
| CNN-LSTM | 0.491 ± 0.036 | 0.479 ± 0.030 | 0.481 ± 0.031 |

These scores are means ± std over 5 folds grouped by recording session, so no session is ever in both training and test. A permutation test for CSP + SVM gives p = 0.29.

## The data

| | |
|---|---|
| Trials | 2160 CSV files, `cellula_MI_data_<n>.csv` |
| Channels | FZ, C3, CZ, C4 (plus a `Time` column) |
| Length | 2500 samples = 10 s at 250 Hz |
| Labels | `labels.csv`: one column `label` (LEFT / RIGHT), 2400 rows, no file names |
| After cleaning | 1831 trials (LEFT 916 / RIGHT 915) from 78 recording sessions |

The notebook pairs label row *i* with file *i*, because the label file gives nothing better. We also tried two other pairings, and neither beats shuffled labels:

- **One trial lost per block.** The files come in 240 blocks of 9 back-to-back trials, while 2400 = 240 × 10. We dropped one label per block, trying each of the 10 positions.
- **Recording time.** We ordered the files by recording time.

## Notebook stages

| Stage | What it does | Main output |
|---|---|---|
| **1. Data** | Reads every trial and checks channels, length, sampling rate and NaN values. Builds the label map. | `outputs/label_map.csv` |
| **2. Processing** | Band-pass 0.5–40 Hz, 50 Hz notch, artifact rejection (269 trials, 12.5 %), CAR comparison, 1–9 s epochs, baseline, normalization. | `outputs/epochs_noCAR.npy`, `epochs_CAR.npy`, `labels.npy`, `removed_trials.csv` |
| **3. EDA** | Class balance, time plots, PSD and ERD on C3/C4, band power. 3.4 covers data quality: sessions, duplicate files, label shifts. | `outputs/duplicates.csv` |
| **4. Features** | CSP log-variance features (8–30 Hz) and PyTorch DataLoaders. | `outputs/csp_features_fold1.npy` |
| **5. Modeling** | CSP + SVM, EEGNet and CNN-LSTM, all on the same session-grouped folds. | `outputs/cv_folds.json` |
| **6. Evaluation** | Accuracy, balanced accuracy, F1, AUC, confusion matrices, ROC, permutation tests. | `outputs/final_results.csv`, `confusion_matrices_roc.png` |
| **7. Deployment** | Trains the final models on all trials, saves and versions them, and checks that the app gives the same result as the notebook. | `models/` |

Each step in the notebook starts with what the task asks (**Task says**), then what the code does and why (**Did**, **Why**), then what the output shows (**Noticed**).

## Folder layout

```text
EEG_Imagery_Classification/
├── labels.csv                      # given labels (no file names)
├── MI-.../MI/MI CSV/*.csv          # the 2160 raw trials
└── final_pipeline/                 # ← this folder
    ├── BCI_MI_Pipeline.ipynb       # the full pipeline (stages 1–7)
    ├── bci_pipeline.py             # shared preprocessing, models, training and inference
    ├── config.json                 # the one config: rate, filters, epoch window, seed, CV folds
    ├── app.py                      # Streamlit entry point (navigation + styling)
    ├── app_core.py                 # app helpers: cached loading, brain-rhythm math, brain drawing
    ├── app_pages/                  # demo.py, models.py, data.py
    ├── .streamlit/config.toml      # app theme
    ├── example_trials/             # one LEFT and one RIGHT trial for the demo
    ├── models/                     # csp_svm.joblib, eegnet.pt, cnn_lstm.pt, model_card.json
    ├── outputs/                    # arrays, tables and figures written by the notebook
    └── project_explained.html      # plain-language explanation of the project
```

`config.json` locates the data relative to this folder: `../MI-*/**/cellula_MI_data_*.csv` and `../labels.csv`.

## How to run

**1. Install** (tested with Python 3.14):

```bash
pip install numpy pandas scipy scikit-learn mne torch matplotlib joblib streamlit altair
pip install openpyxl   # only needed to upload Excel trials in the app
```

**2. Run the notebook.** Open `BCI_MI_Pipeline.ipynb` from this folder and run all cells. It writes `outputs/` and `models/`. The deep-learning cross-validation and permutation tests take several minutes on a CPU.

**3. Start the demo:**

```bash
streamlit run app.py
```

Then open <http://localhost:8501>. The first load takes about 20 s while PyTorch and the models load; after that the app responds instantly. Notebook section 7.3 can also start the app and test it with Streamlit's `AppTest`.

## The demo app

| Page | What you see |
|---|---|
| **Live demo** | Pick a dataset trial (or *Surprise me*), an example trial, or upload a CSV/Excel file. Guess LEFT or RIGHT, then press **Read the brain**. The page shows each model's LEFT/RIGHT with its P(RIGHT), an animated head with C3/C4 glowing by their mu power, a hand that moves with the models' vote, a scoreboard, and tabs for brain rhythms, spectrum, signal and quality check. |
| **Model face-off** | AUC with error bars for the three models, training vs unseen-session accuracy, and why each model ended where it did. |
| **Why it's a coin flip** | The label evidence: LEFT vs RIGHT distributions, the duplicate pairs, and the pairings we tried. |

A model with P(RIGHT) between 0.45 and 0.55 counts as **undecided**.

**Uploaded files** need the columns `FZ, C3, CZ, C4` and at least 2500 rows. Any file from the dataset works, and the upload view has a **Download a sample file** button.

**A direct link** opens one trial already revealed: `http://localhost:8501/?trial=42&guess=LEFT&reveal=1`. Trial 42 is a good one to show, because it is a byte-identical copy of trial 1077, which has the opposite label.

## Design choices forced by the data

- **No cue marker in the files.** Epochs take 1–9 s of each 10 s trial, which skips the filter edges, instead of a window after the cue.
- **No rest period before the imagery.** Baseline correction centers each epoch on its own mean.
- **No subject IDs.** Cross-validation groups trials by recording session instead, using gaps in the `Time` column.
- **Only 4 electrodes.** CAR is computed and compared, but the main models use the signal without CAR, because an average over 4 channels is a poor reference.

## Next steps

1. **Validate the pipeline on data with trusted labels.** BCI Competition IV 2b (MOABB `BNCI2014_004`) has the same C3/Cz/C4 at 250 Hz and the same LEFT vs RIGHT task.
2. **Re-run the notebook** once a label file that names each trial's file exists. The app picks up the new models and model card with no code change.
3. **Put the demo online** (Hugging Face Spaces or Streamlit Community Cloud), with a smaller reference file and ONNX models so the app does not need PyTorch.
4. **Build a real-time demo** that runs a model on sliding windows of a replayed or live recording.
5. **Explain the models** with CSP spatial patterns and EEGNet filters drawn as scalp maps.
