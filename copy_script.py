import os
import shutil

source_directory = #
destination_directory = #
num_files_to_copy = 10000

os.makedirs(destination_directory, exist_ok=True)

jpg_files = [file for file in os.listdir(source_directory) if file.lower().endswith('.jpg')]

for i in range(min(num_files_to_copy, len(jpg_files))):
    source_file = os.path.join(source_directory, jpg_files[i])
    destination_file = os.path.join(destination_directory, jpg_files[i])
    shutil.copy2(source_file, destination_file)

print(f'{num_files_to_copy} jpgs copied')
