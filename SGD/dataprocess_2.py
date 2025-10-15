import os


base_path = r"D:\Projects\N_Caltech101\cy6cvx3ryv-1\Caltech101"


folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
folders.sort() 


if len(folders) != 101:
    raise ValueError(f"Expected 101 folders, found {len(folders)}")

for i, folder in enumerate(folders):
    old_path = os.path.join(base_path, folder)
    temp_path = os.path.join(base_path, f"temp_{i:03d}")
    os.rename(old_path, temp_path)


for i in range(101):
    temp_path = os.path.join(base_path, f"temp_{i:03d}")
    new_path = os.path.join(base_path, str(i))
    os.rename(temp_path, new_path)


print("Folders renamed successfully from 0 to 100.")
