import os
import shutil
import random
from pathlib import Path

# 原始数据路径（每类是0-100编号文件夹，里面是 .bin 文件）
source_root = Path(r"D:\Projects\N_Caltech101\cy6cvx3ryv-1\Caltech101")

# 拆分后保存路径
target_root = Path(r"D:\Projects\N_Caltech101\Caltech101_split")
train_dir = target_root / "train"
test_dir = target_root / "test"

# 创建训练和测试的文件夹（每类一个子目录）
for i in range(101):
    os.makedirs(train_dir / str(i), exist_ok=True)
    os.makedirs(test_dir / str(i), exist_ok=True)

# 遍历每个类别并拆分文件
for class_id in range(101):
    class_path = source_root / str(class_id)
    bin_files = [f for f in os.listdir(class_path) if f.endswith(".bin")]
    random.shuffle(bin_files)

    split_idx = int(len(bin_files) * 0.8)
    train_files = bin_files[:split_idx]
    test_files = bin_files[split_idx:]

    # 拷贝训练 .bin 文件
    for f in train_files:
        shutil.copy(class_path / f, train_dir / str(class_id) / f)

    # 拷贝测试 .bin 文件
    for f in test_files:
        shutil.copy(class_path / f, test_dir / str(class_id) / f)

print("Dataset split (for .bin files) completed successfully.")
