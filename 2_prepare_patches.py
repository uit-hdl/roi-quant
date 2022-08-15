
from glob import glob
import os
import shutil

def rm_n_mkdir(dir_path):
    '''
    Remove (if was present) and create new directory.

    Parameters:
    - dir_path (str):       full path of the directory

    Returns: None
    '''
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path)



if __name__ == "__main__":
    patches_png = sorted(glob("path_to_extracted_patches_from_1_step/*.png"))
    qc_input_patches = "path_to_save_folder/Images/"
    rm_n_mkdir(qc_input_patches)
    
    rename_dict = {}
    for patch_png in patches_png:
        patch_name = os.path.basename(patch_png)
        poly_name = os.path.basename(os.path.dirname(patch_png))
        wsi_name = os.path.basename(os.path.dirname(os.path.dirname(patch_png)))
        rename_dict[patch_png] = f"{wsi_name}~{poly_name}~{patch_name}"
    
    for old_path, new_name in rename_dict.items():
        shutil.copy2(old_path, os.path.join(qc_input_patches, new_name))