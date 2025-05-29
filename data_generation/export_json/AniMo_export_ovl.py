"""
Game Version=Build ID 14307171

[Steam] -> [Ovl files]
"""
################################################################################
root_dir = r"D:\softwares\steam\steamapps\common\[Replace with your game]\win64\ovldata"  # <---- Replace yourself
export_dir = r"D:\xxx\export_ovl_loop"  # <------ Replace yourself
################################################################################

import os
from pathlib import Path
from modules import walker
from generated.formats.ovl import OvlFile
from generated.formats.ovl_base.enums.Compression import Compression
import time

def extract_assets_from_ovl(ovl_path, export_dir, only_types=[]):
    ovl_data = OvlFile()
    ovl_data.load(ovl_path)
    ovl_data.extract(export_dir, only_types=only_types)
    
def keep_specific_files(export_dir):
    for file in os.listdir(export_dir):
        file_path = os.path.join(export_dir, file)
        if os.path.isfile(file_path):
            # Check if the file extension is not .ms2 or .manis
            if not (file.endswith('.ms2') or file.endswith('.manis')):
                # Remove the file
                os.remove(file_path)
                print(f"Removed: {file_path}")
                
def find_and_extract_ovls(root_dir, export_dir, filter_subdir="Animals"):
    cnt = 0
    for root, _, files in os.walk(root_dir):
        if any(filter_subdir in part for part in Path(root).parts):
            for file in files:
                if file.endswith('.ovl'):
                    if 'AssetPackagesExtrasList' in file:
                        continue
                    ovl_path = os.path.join(root, file)
                    print(f"Extracting assets from {ovl_path}")
                    cnt += 1
                    extract_assets_from_ovl(ovl_path, os.path.join(export_dir,file))
                    keep_specific_files(os.path.join(export_dir,file))
    print(f"extract {cnt} ovl files")
    if cnt == 0:
        print("Is your path correct","?"*100)
        print(f"\t root_dir:{root_dir}","<","-"*40,"Replace to your own path")                
        print(f"\t export_dir:{export_dir}","<","-"*40,"Replace to your own path")                
    
if __name__ == "__main__":
    # Ensure the export directory exists
    Path(export_dir).mkdir(parents=True, exist_ok=True)
    find_and_extract_ovls(root_dir, export_dir, filter_subdir="Animals")
    
