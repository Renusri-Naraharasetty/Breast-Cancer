# BCMID Dataset Directory

The dataset directory should be organized patient-wise:
```
BCMID/
├── <patient_id>/
│   ├── Mammogram/
│   │   ├── <image_1>.png
│   │   └── ...
│   ├── Ultrasound/
│   │   ├── <image_1>.png
│   │   └── ...
│   └── <patient_id>_US_MG_Report.pdf
└── ...
```

By default, the dataset path is configured in `configs/config.yaml`:
```yaml
data:
  dataset_dir: "BCMID"
```
