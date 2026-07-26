# -*- coding: utf-8 -*-
"""
YOLO-style multi-label spine MRI dataset profiler.

Outputs:
- dataset_summary.json / dataset_summary.csv
- class_distribution.csv
- multilabel_combinations.csv
- image_statistics.csv / box_statistics.csv
- patient_split_summary.csv
- data_quality_issues.csv
- README_DATASET_STATISTICS.md
- figures/*.png

Label format:
<class_ids> <cx> <cy> <w> <h>
Example: 0,2 0.512 0.486 0.238 0.164
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CFG:
    ROOT = "/home/ads_lj/visiontask/datasets_yolo_8_1_1"
    OUTPUT_DIR = "/home/ads_lj/visiontask/datasets_yolo_8_1_1/dataset_profile"
    SPLITS = ["train", "val", "test"]
    CLASS_NAMES = [
        "VCF", "Old VCF", "Cement", "Fixation", "Hemangioma",
        "Malignant", "Schmorl's Node", "Kummell's Disease", "Infection",
    ]
    IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

    # Patient ID options:
    # 1) mapping CSV columns: image_path, patient_id
    PATIENT_MAPPING_CSV = None
    # 2) regex, first captured group is patient ID. Example: r"(R\d{9})"
    PATIENT_ID_REGEX = None
    # 3) images/train/PATIENT_ID/image.jpg structure
    FIRST_SUBDIRECTORY_IS_PATIENT = False

    TOP_COMBINATIONS = 20
    README_FIGURE_PREFIX = "dataset_profile/figures"


def stats(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    if len(a) == 0:
        return {k: None for k in ["mean", "std", "min", "q1", "median", "q3", "max"]}
    return {
        "mean": float(a.mean()), "std": float(a.std()), "min": float(a.min()),
        "q1": float(np.percentile(a, 25)), "median": float(np.median(a)),
        "q3": float(np.percentile(a, 75)), "max": float(a.max()),
    }


def load_patient_mapping(path):
    if not path:
        return {}
    df = pd.read_csv(path)
    if not {"image_path", "patient_id"}.issubset(df.columns):
        raise ValueError("Patient mapping CSV needs image_path and patient_id columns.")
    mapping = {}
    for _, r in df.iterrows():
        key = str(r["image_path"]).replace("\\", "/").lstrip("./")
        pid = str(r["patient_id"]).strip()
        mapping[key] = pid
        mapping[Path(key).name] = pid
    return mapping


def get_patient_id(image_path, split_dir, split, mapping, regex, first_subdir):
    rel = image_path.relative_to(split_dir)
    keys = [
        str(Path("images") / split / rel).replace("\\", "/"),
        str(Path(split) / rel).replace("\\", "/"),
        str(rel).replace("\\", "/"),
        image_path.name,
    ]
    for key in keys:
        if key in mapping:
            return mapping[key]
    if regex:
        m = re.search(regex, str(rel).replace("\\", "/"))
        if m:
            return m.group(1) if m.groups() else m.group(0)
    if first_subdir and len(rel.parts) > 1:
        return rel.parts[0]
    return None


def parse_line(line, num_classes):
    p = line.split()
    if len(p) < 5:
        return None, "fewer_than_5_fields"
    try:
        classes = sorted(set(int(x) for x in p[0].split(",") if x != ""))
        cx, cy, w, h = map(float, p[1:5])
    except Exception:
        return None, "parse_error"
    if not classes:
        return None, "empty_class_set"
    if any(c < 0 or c >= num_classes for c in classes):
        return None, "class_id_out_of_range"
    if not np.isfinite([cx, cy, w, h]).all():
        return None, "non_finite_coordinate"
    if w <= 0 or h <= 0:
        return None, "non_positive_box_size"

    x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
    issues = []
    if any(v < 0 or v > 1 for v in [cx, cy, w, h]):
        issues.append("normalized_coordinate_out_of_range")
    if x1 < 0 or y1 < 0 or x2 > 1 or y2 > 1:
        issues.append("box_extends_outside_image")

    return {
        "classes": classes, "cx": cx, "cy": cy, "w": w, "h": h,
        "area": w*h, "aspect_ratio": w/h, "issues": issues,
    }, None


def scan(root, mapping_csv=None, regex=None, first_subdir=False):
    mapping = load_patient_mapping(mapping_csv)
    images, boxes, issues = [], [], []
    patient_splits = defaultdict(set)
    nclasses = len(CFG.CLASS_NAMES)

    for split in CFG.SPLITS:
        idir, ldir = root/"images"/split, root/"labels"/split
        image_paths = sorted(
            p for p in idir.rglob("*")
            if p.is_file() and p.suffix.lower() in CFG.IMAGE_EXTENSIONS
        )

        image_stems = {p.relative_to(idir).with_suffix("") for p in image_paths}
        for lp in sorted(ldir.rglob("*.txt")):
            if lp.relative_to(ldir).with_suffix("") not in image_stems:
                issues.append({
                    "split": split, "image_path": "", "label_path": str(lp),
                    "issue_type": "label_without_matching_image", "details": "",
                })

        for ip in image_paths:
            rel = ip.relative_to(idir)
            lp = (ldir/rel).with_suffix(".txt")
            pid = get_patient_id(ip, idir, split, mapping, regex, first_subdir)
            if pid:
                patient_splits[pid].add(split)

            width = height = None
            mode = None
            try:
                with Image.open(ip) as im:
                    width, height = im.size
                    mode = im.mode
            except Exception as e:
                issues.append({
                    "split": split, "image_path": str(ip), "label_path": str(lp),
                    "issue_type": "image_read_error", "details": repr(e),
                })

            num_boxes = total_labels = multilabel_boxes = 0
            unique_classes = set()

            if not lp.exists():
                issues.append({
                    "split": split, "image_path": str(ip), "label_path": str(lp),
                    "issue_type": "image_without_matching_label", "details": "",
                })
            else:
                lines = [x.strip() for x in lp.read_text(
                    encoding="utf-8", errors="replace").splitlines() if x.strip()]
                if not lines:
                    issues.append({
                        "split": split, "image_path": str(ip), "label_path": str(lp),
                        "issue_type": "empty_label_file", "details": "",
                    })

                for duplicate, count in Counter(lines).items():
                    if count > 1:
                        issues.append({
                            "split": split, "image_path": str(ip), "label_path": str(lp),
                            "issue_type": "duplicate_annotation_line",
                            "details": f"count={count}; {duplicate}",
                        })

                for line_no, line in enumerate(lines, 1):
                    parsed, error = parse_line(line, nclasses)
                    if error:
                        issues.append({
                            "split": split, "image_path": str(ip), "label_path": str(lp),
                            "issue_type": error, "details": f"line={line_no}; {line}",
                        })
                        continue
                    for issue in parsed["issues"]:
                        issues.append({
                            "split": split, "image_path": str(ip), "label_path": str(lp),
                            "issue_type": issue, "details": f"line={line_no}; {line}",
                        })

                    cls = parsed["classes"]
                    names = " + ".join(CFG.CLASS_NAMES[c] for c in cls)
                    num_boxes += 1
                    total_labels += len(cls)
                    multilabel_boxes += int(len(cls) > 1)
                    unique_classes.update(cls)

                    boxes.append({
                        "split": split, "image_path": str(ip),
                        "relative_image_path": str(Path("images")/split/rel).replace("\\", "/"),
                        "patient_id": pid, "class_ids": ",".join(map(str, cls)),
                        "class_names": names, "num_labels": len(cls),
                        "cx": parsed["cx"], "cy": parsed["cy"],
                        "width_normalized": parsed["w"],
                        "height_normalized": parsed["h"],
                        "area_normalized": parsed["area"],
                        "aspect_ratio": parsed["aspect_ratio"],
                        "box_width_pixels": parsed["w"]*width if width else None,
                        "box_height_pixels": parsed["h"]*height if height else None,
                    })

            images.append({
                "split": split, "image_path": str(ip),
                "relative_image_path": str(Path("images")/split/rel).replace("\\", "/"),
                "patient_id": pid, "image_width": width, "image_height": height,
                "image_mode": mode, "num_boxes": num_boxes,
                "total_labels": total_labels, "num_unique_classes": len(unique_classes),
                "num_multilabel_boxes": multilabel_boxes,
                "has_annotation": int(num_boxes > 0),
            })

    return pd.DataFrame(images), pd.DataFrame(boxes), pd.DataFrame(issues), patient_splits


def summarize(image_df, box_df, patient_splits):
    split_rows = []
    for split in CFG.SPLITS:
        im = image_df[image_df.split == split]
        bx = box_df[box_df.split == split]
        split_rows.append({
            "split": split, "num_images": len(im),
            "num_patients": im.patient_id.dropna().nunique(),
            "num_boxes": len(bx),
            "num_labels": int(bx.num_labels.sum()) if len(bx) else 0,
            "num_multilabel_boxes": int((bx.num_labels > 1).sum()) if len(bx) else 0,
            "mean_boxes_per_image": float(im.num_boxes.mean()) if len(im) else 0,
            "mean_labels_per_box": float(bx.num_labels.mean()) if len(bx) else 0,
        })
    split_df = pd.DataFrame(split_rows)

    class_rows = []
    for c, name in enumerate(CFG.CLASS_NAMES):
        for split in CFG.SPLITS + ["all"]:
            d = box_df if split == "all" else box_df[box_df.split == split]
            if len(d):
                mask = d.class_ids.astype(str).apply(
                    lambda s: c in [int(x) for x in s.split(",") if x != ""]
                )
                x = d[mask]
            else:
                x = d
            class_rows.append({
                "split": split, "class_id": c, "class_name": name,
                "box_count": len(x), "image_count": x.image_path.nunique(),
                "patient_count": x.patient_id.dropna().nunique(),
            })
    class_df = pd.DataFrame(class_rows)

    if len(box_df):
        combo_df = (
            box_df.groupby(["class_ids", "class_names"])
            .agg(box_count=("class_ids", "size"),
                 image_count=("image_path", "nunique"),
                 patient_count=("patient_id", lambda x: x.dropna().nunique()))
            .reset_index().sort_values("box_count", ascending=False)
        )
        combo_df["percentage_of_boxes"] = combo_df.box_count / len(box_df) * 100
    else:
        combo_df = pd.DataFrame(columns=[
            "class_ids", "class_names", "box_count", "image_count",
            "patient_count", "percentage_of_boxes",
        ])

    patient_df = pd.DataFrame([
        {"patient_id": pid, "splits": ",".join(sorted(splits)),
         "num_splits": len(splits), "has_split_leakage": int(len(splits) > 1)}
        for pid, splits in sorted(patient_splits.items())
    ])

    summary = {
        "total_images": int(len(image_df)),
        "total_boxes": int(len(box_df)),
        "total_labels": int(box_df.num_labels.sum()) if len(box_df) else 0,
        "unique_patients": int(image_df.patient_id.dropna().nunique()),
        "patient_count_available": bool(image_df.patient_id.notna().any()),
        "patients_in_multiple_splits": int(patient_df.has_split_leakage.sum())
        if len(patient_df) else 0,
        "multilabel_boxes": int((box_df.num_labels > 1).sum()) if len(box_df) else 0,
        "multilabel_box_percentage": float((box_df.num_labels > 1).mean()*100)
        if len(box_df) else 0,
        "images_without_valid_boxes": int((image_df.num_boxes == 0).sum()),
        "boxes_per_image": stats(image_df.num_boxes),
        "labels_per_box": stats(box_df.num_labels if len(box_df) else []),
        "box_width_normalized": stats(box_df.width_normalized if len(box_df) else []),
        "box_height_normalized": stats(box_df.height_normalized if len(box_df) else []),
        "box_area_normalized": stats(box_df.area_normalized if len(box_df) else []),
        "box_aspect_ratio": stats(box_df.aspect_ratio if len(box_df) else []),
    }
    return summary, split_df, class_df, combo_df, patient_df


def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plots(out, image_df, box_df, split_df, class_df, combo_df):
    fdir = out/"figures"
    fdir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 5))
    plt.bar(split_df.split, split_df.num_images)
    plt.title("Number of Images by Dataset Split")
    plt.ylabel("Images")
    savefig(fdir/"01_split_image_counts.png")

    allc = class_df[class_df.split == "all"].sort_values("box_count")
    plt.figure(figsize=(10, 6))
    plt.barh(allc.class_name, allc.box_count)
    plt.title("Class Distribution by Bounding Box Count")
    plt.xlabel("Bounding Boxes")
    savefig(fdir/"02_class_box_distribution.png")

    allc = class_df[class_df.split == "all"].sort_values("image_count")
    plt.figure(figsize=(10, 6))
    plt.barh(allc.class_name, allc.image_count)
    plt.title("Class Distribution by Image Count")
    plt.xlabel("Images")
    savefig(fdir/"03_class_image_distribution.png")

    plt.figure(figsize=(8, 5))
    plt.hist(image_df.num_boxes, bins=np.arange(-0.5, image_df.num_boxes.max()+1.5, 1))
    plt.title("Bounding Boxes per Image")
    plt.xlabel("Boxes per Image")
    plt.ylabel("Images")
    savefig(fdir/"04_boxes_per_image.png")

    plt.figure(figsize=(8, 5))
    if len(box_df):
        plt.hist(box_df.num_labels, bins=np.arange(0.5, box_df.num_labels.max()+1.5, 1))
    plt.title("Labels per Bounding Box")
    plt.xlabel("Labels")
    plt.ylabel("Boxes")
    savefig(fdir/"05_labels_per_box.png")

    plt.figure(figsize=(8, 5))
    if len(box_df):
        plt.hist(box_df.area_normalized, bins=50)
    plt.title("Normalized Bounding Box Area")
    plt.xlabel("Width × Height")
    plt.ylabel("Boxes")
    savefig(fdir/"06_box_area_distribution.png")

    plt.figure(figsize=(7, 6))
    if len(box_df):
        d = box_df.sample(min(len(box_df), 30000), random_state=42)
        plt.scatter(d.width_normalized, d.height_normalized, s=8, alpha=0.3)
    plt.xlim(0, 1); plt.ylim(0, 1)
    plt.title("Bounding Box Width vs Height")
    plt.xlabel("Normalized Width"); plt.ylabel("Normalized Height")
    savefig(fdir/"07_box_width_height.png")

    plt.figure(figsize=(8, 5))
    if len(box_df):
        a = box_df.aspect_ratio.replace([np.inf, -np.inf], np.nan).dropna()
        a = a[a <= np.percentile(a, 99)]
        plt.hist(a, bins=50)
    plt.title("Bounding Box Aspect Ratio")
    plt.xlabel("Width / Height"); plt.ylabel("Boxes")
    savefig(fdir/"08_box_aspect_ratio.png")

    top = combo_df.head(CFG.TOP_COMBINATIONS).sort_values("box_count")
    plt.figure(figsize=(12, 8))
    if len(top):
        plt.barh(top.class_names, top.box_count)
    plt.title(f"Top {CFG.TOP_COMBINATIONS} Label Combinations")
    plt.xlabel("Boxes")
    savefig(fdir/"09_top_multilabel_combinations.png")

    matrix = np.zeros((len(CFG.CLASS_NAMES), len(CFG.CLASS_NAMES)), dtype=int)
    for s in box_df.class_ids.astype(str) if len(box_df) else []:
        ids = [int(x) for x in s.split(",")]
        for a in ids:
            for b in ids:
                matrix[a, b] += 1
    plt.figure(figsize=(10, 8))
    im = plt.imshow(matrix, aspect="auto")
    plt.colorbar(im, label="Count")
    plt.xticks(range(len(CFG.CLASS_NAMES)), CFG.CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CFG.CLASS_NAMES)), CFG.CLASS_NAMES)
    plt.title("Class Co-occurrence Matrix")
    savefig(fdir/"10_class_cooccurrence_heatmap.png")

    plt.figure(figsize=(7, 5))
    if split_df.num_patients.sum() > 0:
        plt.bar(split_df.split, split_df.num_patients)
        plt.title("Number of Patients by Split")
        plt.ylabel("Patients")
    else:
        plt.text(0.5, 0.5, "Patient IDs unavailable.\nConfigure mapping CSV or regex.",
                 ha="center", va="center")
        plt.axis("off")
    savefig(fdir/"11_patient_counts_by_split.png")

    dims = image_df.dropna(subset=["image_width", "image_height"])
    plt.figure(figsize=(8, 6))
    if len(dims):
        x = dims.groupby(["image_width", "image_height"]).size().reset_index(name="count")
        x = x.sort_values("count", ascending=False).head(30)
        labels = [f"{int(w)}×{int(h)}" for w, h in zip(x.image_width, x.image_height)]
        plt.barh(labels[::-1], x["count"][::-1])
        plt.title("Most Common Original Image Dimensions")
        plt.xlabel("Images")
    savefig(fdir/"12_image_size_distribution.png")


def md_table(df, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"]*len(cols)) + " |"]
    for _, r in df[cols].iterrows():
        vals = []
        for c in cols:
            v = r[c]
            vals.append(f"{v:.4f}" if isinstance(v, float) and np.isfinite(v) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_markdown(out, summary, split_df, class_df, combo_df, issue_count):
    class_all = class_df[class_df.split == "all"]
    combo = combo_df.head(15).copy()
    if len(combo):
        combo["percentage_of_boxes"] = combo.percentage_of_boxes.map(lambda x: f"{x:.2f}%")

    def n(v):
        return 0 if v is None else v

    patient_value = f"{summary['unique_patients']:,}" if summary["patient_count_available"] else "Not available"
    p = CFG.README_FIGURE_PREFIX

    text = f"""## Dataset Statistics

### Overall Summary

| Metric | Value |
|---|---:|
| Total images | {summary['total_images']:,} |
| Total bounding boxes | {summary['total_boxes']:,} |
| Total class labels | {summary['total_labels']:,} |
| Unique patients | {patient_value} |
| Multi-label bounding boxes | {summary['multilabel_boxes']:,} |
| Multi-label box proportion | {summary['multilabel_box_percentage']:.2f}% |
| Images without valid boxes | {summary['images_without_valid_boxes']:,} |
| Patients appearing in multiple splits | {summary['patients_in_multiple_splits']:,} |
| Data quality issues detected | {issue_count:,} |

### Dataset Split

{md_table(split_df, ["split", "num_images", "num_patients", "num_boxes", "num_labels",
                     "num_multilabel_boxes", "mean_boxes_per_image", "mean_labels_per_box"])}

![Dataset split]({p}/01_split_image_counts.png)

![Patient counts]({p}/11_patient_counts_by_split.png)

### Class Distribution

{md_table(class_all, ["class_id", "class_name", "box_count", "image_count", "patient_count"])}

![Class box distribution]({p}/02_class_box_distribution.png)

![Class image distribution]({p}/03_class_image_distribution.png)

### Bounding Box Statistics

| Metric | Mean | Median | Q1 | Q3 | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Normalized width | {n(summary['box_width_normalized']['mean']):.4f} | {n(summary['box_width_normalized']['median']):.4f} | {n(summary['box_width_normalized']['q1']):.4f} | {n(summary['box_width_normalized']['q3']):.4f} | {n(summary['box_width_normalized']['min']):.4f} | {n(summary['box_width_normalized']['max']):.4f} |
| Normalized height | {n(summary['box_height_normalized']['mean']):.4f} | {n(summary['box_height_normalized']['median']):.4f} | {n(summary['box_height_normalized']['q1']):.4f} | {n(summary['box_height_normalized']['q3']):.4f} | {n(summary['box_height_normalized']['min']):.4f} | {n(summary['box_height_normalized']['max']):.4f} |
| Normalized area | {n(summary['box_area_normalized']['mean']):.4f} | {n(summary['box_area_normalized']['median']):.4f} | {n(summary['box_area_normalized']['q1']):.4f} | {n(summary['box_area_normalized']['q3']):.4f} | {n(summary['box_area_normalized']['min']):.4f} | {n(summary['box_area_normalized']['max']):.4f} |
| Aspect ratio | {n(summary['box_aspect_ratio']['mean']):.4f} | {n(summary['box_aspect_ratio']['median']):.4f} | {n(summary['box_aspect_ratio']['q1']):.4f} | {n(summary['box_aspect_ratio']['q3']):.4f} | {n(summary['box_aspect_ratio']['min']):.4f} | {n(summary['box_aspect_ratio']['max']):.4f} |

![Boxes per image]({p}/04_boxes_per_image.png)

![Labels per box]({p}/05_labels_per_box.png)

![Box area]({p}/06_box_area_distribution.png)

![Box width and height]({p}/07_box_width_height.png)

![Box aspect ratio]({p}/08_box_aspect_ratio.png)

### Multi-Label Combinations

{md_table(combo, ["class_ids", "class_names", "box_count", "image_count",
                  "patient_count", "percentage_of_boxes"]) if len(combo) else "No valid combinations found."}

![Top label combinations]({p}/09_top_multilabel_combinations.png)

![Class co-occurrence]({p}/10_class_cooccurrence_heatmap.png)

### Image Dimensions

![Image dimensions]({p}/12_image_size_distribution.png)

### Data Quality Control

The automated checks include missing image-label pairs, empty label files, invalid
class IDs, malformed coordinates, boxes outside image boundaries, duplicate
annotation rows, and patient overlap across dataset splits.

Detailed results:

```text
dataset_profile/data_quality_issues.csv
```
"""
    path = out/"README_DATASET_STATISTICS.md"
    path.write_text(text, encoding="utf-8")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=CFG.ROOT)
    ap.add_argument("--output-dir", default=CFG.OUTPUT_DIR)
    ap.add_argument("--patient-mapping-csv", default=CFG.PATIENT_MAPPING_CSV)
    ap.add_argument("--patient-id-regex", default=CFG.PATIENT_ID_REGEX)
    ap.add_argument("--first-subdirectory-is-patient", action="store_true",
                    default=CFG.FIRST_SUBDIRECTORY_IS_PATIENT)
    args = ap.parse_args()

    root = Path(args.root).expanduser().resolve()
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)

    image_df, box_df, issue_df, patient_splits = scan(
        root, args.patient_mapping_csv, args.patient_id_regex,
        args.first_subdirectory_is_patient
    )
    summary, split_df, class_df, combo_df, patient_df = summarize(
        image_df, box_df, patient_splits
    )

    image_df.to_csv(out/"image_statistics.csv", index=False, encoding="utf-8-sig")
    box_df.to_csv(out/"box_statistics.csv", index=False, encoding="utf-8-sig")
    issue_df.to_csv(out/"data_quality_issues.csv", index=False, encoding="utf-8-sig")
    split_df.to_csv(out/"dataset_summary.csv", index=False, encoding="utf-8-sig")
    class_df.to_csv(out/"class_distribution.csv", index=False, encoding="utf-8-sig")
    combo_df.to_csv(out/"multilabel_combinations.csv", index=False, encoding="utf-8-sig")
    patient_df.to_csv(out/"patient_split_summary.csv", index=False, encoding="utf-8-sig")
    with (out/"dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plots(out, image_df, box_df, split_df, class_df, combo_df)
    readme = generate_markdown(out, summary, split_df, class_df, combo_df, len(issue_df))

    print("="*80)
    print(f"Images: {summary['total_images']:,}")
    print(f"Boxes: {summary['total_boxes']:,}")
    print(f"Labels: {summary['total_labels']:,}")
    print(f"Patients: {summary['unique_patients'] if summary['patient_count_available'] else 'unavailable'}")
    print(f"Patient split leakage: {summary['patients_in_multiple_splits']:,}")
    print(f"Quality issues: {len(issue_df):,}")
    print(f"Output: {out}")
    print(f"README section: {readme}")


if __name__ == "__main__":
    main()
