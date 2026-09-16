# Competition brief

Verified September 16, 2026 using official Kaggle documentation, the authenticated Kaggle API and the five downloaded CSVs.

## Task and execution

Predict 12 probabilities per knee MRI study: ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion and Fracture. The submission ID is `StudyInstanceUID`.

The [official evaluation page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation) specifies the mean of the 12 ROC AUC scores, with larger values better. Submissions run as Kaggle notebooks, with internet disabled and a maximum of nine hours on CPU or GPU, producing `submission.csv`. Public external data and pretrained models are allowed under the competition rules; check the specific source and full rules before incorporating either.

Our metric helper requires both classes for every target and refuses to silently drop an undefined AUC. Predictions are probabilities in [0, 1], not thresholded diagnoses. The local validator also requires sample order, a deliberate consistency check.

The API confirms notebook-only submissions, five submissions per day, entry and team-merger deadlines of October 15, and final submission on October 22, 2026. All deadlines are 23:59 UTC. Recheck official pages before relying on a deadline or rule.

## Data actually downloaded

| File | Rows | Contents |
| --- | ---: | --- |
| train.csv | 4,407 | Study ID, Report, 12 condition columns |
| train_series.csv | 24,371 | Study/series IDs, fluid sensitivity, fat suppression, anatomical plane |
| test.csv | 3 | Study IDs only |
| test_series.csv | 15 | Same series metadata schema |
| sample_submission.csv | 3 | Study ID and 12 probability columns |

Only **58 training rows have complete explicit labels**; 4,349 have none, and none are partially labeled in this snapshot. All 4,407 have reports. Those reports are **absent at test time**. Keep unknown conditions missing; derive training labels only through a documented extraction and review process.

No patient ID or site column is present in these CSVs. `StudyInstanceUID` identifies studies, not necessarily independent patients. Inspect DICOM identifiers and duplicates before defining the final validation groups. Do not assume an anonymized DICOM PatientID is reliable across sites.

The [official data page](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) lists roughly **570 GB** of imaging. Images live under `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm` and the analogous test directory. DICOM filenames are identifiers, not slice order. The example test data is replaced during evaluation; the documented full test set is approximately 1,300 studies.

`make download` fetches only the five CSVs (about 9 MB). `make data` records their SHA-256 hashes and aggregate findings under `artifacts/reports/`. Dataset updates may change these counts.

## Sources and remaining work

- [Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview), [evaluation](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation), [data](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data), [rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules).
- [RSNA announcement](https://www.rsna.org/media/press/2026/2669) describes the MRI/report learning task and international, multilingual data.
- The authenticated Kaggle competition-list API independently confirms deadlines, access, the ROC AUC metric family and notebook-only submission mode.

Next verify patient grouping, site/scanner shifts, DICOM decoding, image orientation and usable series, and the exact report-label definitions. Detailed rules and any external label/model source need review before use. No DICOM images or external weights have been downloaded for this scaffold.
