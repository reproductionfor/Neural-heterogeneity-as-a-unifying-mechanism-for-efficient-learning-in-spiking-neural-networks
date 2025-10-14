import os

# 指定路径
base_path = r"D:\Projects\N_Caltech101\cy6cvx3ryv-1\Caltech101"

# 获取所有文件夹并按字母排序
folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
folders.sort()  # 根据名字排序，确保顺序一致

# 检查是否刚好有 101 个文件夹
if len(folders) != 101:
    raise ValueError(f"Expected 101 folders, found {len(folders)}")

# 重命名文件夹（避免命名冲突，先改为临时名）
for i, folder in enumerate(folders):
    old_path = os.path.join(base_path, folder)
    temp_path = os.path.join(base_path, f"temp_{i:03d}")
    os.rename(old_path, temp_path)

# 再从临时名改为目标名
for i in range(101):
    temp_path = os.path.join(base_path, f"temp_{i:03d}")
    new_path = os.path.join(base_path, str(i))
    os.rename(temp_path, new_path)

print("Folders renamed successfully from 0 to 100.")