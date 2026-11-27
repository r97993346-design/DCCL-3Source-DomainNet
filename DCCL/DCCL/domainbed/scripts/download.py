import os
import json
import shutil

IMG_ROOT = "terra_incognita_raw"
ANN_DIR = "eccv_18_annotation_files"
SAVE_ROOT = "terra_incognita"

ANN_FILES = [
    "train_annotations.json",
    "cis_val_annotations.json",
    "cis_test_annotations.json",
    "trans_val_annotations.json",
    "trans_test_annotations.json",
]

os.makedirs(SAVE_ROOT, exist_ok=True)

# 读取所有 annotation
images = []
annotations = []
categories = []

for ann_file in ANN_FILES:
    with open(os.path.join(ANN_DIR, ann_file), "r") as f:
        data = json.load(f)
        images.extend(data["images"])
        annotations.extend(data["annotations"])
        categories = data["categories"]  # 相同

# 构建映射
cat_id_to_name = {c["id"]: c["name"] for c in categories}
img_id_to_info = {img["id"]: img for img in images}

img_to_cat = {}
for ann in annotations:
    if ann["image_id"] not in img_to_cat:
        img_to_cat[ann["image_id"]] = ann["category_id"]

# 建立 filename -> path
filename_to_path = {}
for root, _, files in os.walk(IMG_ROOT):
    for fn in files:
        filename_to_path[fn] = os.path.join(root, fn)

# 拷贝
count = 0
for img_id, cat_id in img_to_cat.items():
    img_info = img_id_to_info[img_id]
    filename = img_info["file_name"]

    if filename not in filename_to_path:
        continue

    src = filename_to_path[filename]

    location = src.split("/")[-2]
    cat_name = cat_id_to_name[cat_id]

    dst_dir = os.path.join(SAVE_ROOT, location, cat_name)
    os.makedirs(dst_dir, exist_ok=True)

    shutil.copy(src, os.path.join(dst_dir, filename))
    count += 1

print("done:", count)