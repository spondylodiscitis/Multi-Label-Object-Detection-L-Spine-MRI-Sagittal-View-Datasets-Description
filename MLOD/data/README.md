# Data directory

The clinical dataset is not included in this repository.

Expected local structure:

```text
datasets_yolo_8_1_1/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/
    ├── train/
    ├── val/
    └── test/
```

Each annotation row uses:

```text
<class_ids> <center_x> <center_y> <width> <height>
```

Multi-label example:

```text
0,2 0.512 0.486 0.238 0.164
```

Never commit clinical images, label files containing sensitive metadata, or
patient mapping tables.
