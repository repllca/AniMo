"""
[Manis & Ms2 files] -> [Json files]

Please Copy & Paste this script into your Blender (3.6 version) script region.
Please Copy & Paste this script into your Blender (3.6 version) script region.
Please Copy & Paste this script into your Blender (3.6 version) script region.
"""
################################################################################
root_dir_kk = r"D:\xxx\export_ovl_loop"      # <---- Replace yourself
output_json_kk = r"D:\xxx\export_json_loop"  # <---- Replace yourself
################################################################################


import bpy
import json
import os
from plugin.modules_import.armature import import_armature, append_armature_modifier, import_vertex_groups, get_bone_names
from plugin.modules_import.geometry import import_mesh_layers, import_shapekeys, ob_postpro, append_mirror_modifier, get_valid_lod_objects, import_mesh_properties
from plugin.modules_import.material import import_material
from plugin.utils.hair import add_psys
from plugin.utils.shell import is_shell, gauge_uv_scale_wrapper
from plugin.utils.object import create_ob, create_scene, create_collection, set_collection_visibility
from generated.formats.ms2 import Ms2File
from plugin import import_fgm, import_ms2, import_spl, import_manis
import bpy
import zipfile


def safe_clear_scene():
    # Clear all objects
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()

    # Clear all collections except the default "Scene Collection"
    for collection in bpy.data.collections:
        if collection.name != "Scene Collection":
            bpy.data.collections.remove(collection)

    # Clear orphaned data
    for datablock in [bpy.data.meshes, bpy.data.materials, bpy.data.textures, 
                      bpy.data.images, bpy.data.armatures, bpy.data.actions]:
        for item in datablock:
            if not item.users:
                datablock.remove(item)

class DefaultReporter:
    def __init__(self):
        self.info_messages = []
        self.warning_messages = []
        self.error_messages = []

    def show_info(self, message):
        print(f"INFO: {message}")
        self.info_messages.append(message)

    def show_warning(self, message):
        print(f"WARNING: {message}")
        self.warning_messages.append(message)

    def show_error(self, message):
        print(f"ERROR: {message}")
        self.error_messages.append(message)

    def __call__(self, message_type, message):
        if message_type == {'INFO'}:
            self.show_info(message)
        elif message_type == {'WARNING'}:
            self.show_warning(message)
        elif message_type == {'ERROR'}:
            self.show_error(message)

kkreport = DefaultReporter()

def load_ms2(filepath: str, use_custom_normals: bool = False, mirror_mesh: bool = False):
    import_ms2.load(reporter=kkreport, filepath=filepath)
    print("[ Ms2 ] File loaded successfully")
    print('='*40)
    print(filepath)
    print('='*40)

def load_manis(filepath: str, use_custom_normals: bool = False, mirror_mesh: bool = False):
    import_manis.load(reporter=kkreport, filepath=filepath)
    print("[Manis] File loaded successfully")
    print('='*40)
    print(filepath)
    print('='*40)

def get_bone_positions(armature, action):
    bone_positions = {}
    
    armature.animation_data.action = action
    
    frame_start = int(action.frame_range[0])
    frame_end = int(action.frame_range[1])
    
    for frame in range(frame_start, frame_end + 1):
        bpy.context.scene.frame_set(frame)
        bone_positions[frame] = {}
        for bone in armature.pose.bones:
            bone_positions[frame][bone.name] = bone.head.xyz[:]
    
    return bone_positions

def export_to_json(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=4)

def import_and_export(ms2_path, manis_path, output_dir):
    try:
        ms2_name = os.path.splitext(os.path.basename(ms2_path))[0]
        manis_name = os.path.splitext(os.path.basename(manis_path))[0]
        
        if 'aardvark_' in ms2_name:
            if 'animationmotionextractedbehaviour' not in manis_name:
                return
        safe_clear_scene()
        load_ms2(ms2_path)
        load_manis(manis_path)
        armature = None
        for obj in bpy.context.scene.objects:
            if obj.type == 'ARMATURE':
                armature = obj
                break

        if armature:
            # Iterate through all actions
            for action in bpy.data.actions:
                if action.id_root == 'OBJECT':  # Check if the action is applicable to objects
                    
                    
                    # Create safe action name
                    safe_action_name = action.name.replace('@', '_')
                    
                    if ms2_name[:-1] in safe_action_name:
                        # Create output filename
                        output_filename = f"{ms2_name}_{manis_name}_{safe_action_name}_keypoints.json"
                        output_path = os.path.join(output_dir, output_filename)
                        bone_positions = get_bone_positions(armature, action)
                        export_to_json(bone_positions, output_path)
                        print(f"Export completed for action '{action.name}' in {manis_path}!")
                    else:
                        print('.', end='')
        else:
            print("No armature found in the scene.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


def zip_directory(directory_path, zip_path):
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, directory_path)
                zipf.write(file_path, arcname)

dir_ls = os.listdir(root_dir_kk)

for dir_name in dir_ls:
    zip_path = os.path.join(output_json_kk, f"{dir_name}.zip")
    input_dir = os.path.join(root_dir_kk, dir_name)
    
    if os.path.exists(zip_path):
        print('exist zip pass')
        continue
    
    # Find the .ms2 file
    ms2_file = None
    for filename in os.listdir(input_dir):
        if filename.endswith(".ms2"):
            ms2_file = os.path.join(input_dir, filename)
            break

    if not ms2_file:
        print(f"No .ms2 file found in the directory {dir_name}.")
        continue
    else:
        # Process all .manis files
        processed = False
        for filename in os.listdir(input_dir):
            if filename.endswith(".manis"):
                manis_path = os.path.join(input_dir, filename)
                output_dir = os.path.join(output_json_kk, dir_name)
                os.makedirs(output_dir, exist_ok=True)
                
                print(f"Processing {filename}...")
                import_and_export(ms2_file, manis_path, output_dir)
                processed = True

        if processed:
            print(f"All animations in {dir_name} processed!")
            
            zip_directory(output_dir, zip_path)
            print(f"Zipped {dir_name} to {zip_path}")
            
            safe_clear_scene()
            
            # Remove the original unzipped directory
            for root, dirs, files in os.walk(output_dir, topdown=False):
                for file in files:
                    os.remove(os.path.join(root, file))
                for dir in dirs:
                    os.rmdir(os.path.join(root, dir))
            try:
                os.rmdir(output_dir)
            except Exception as e:
                pass
            print(f"Removed unzipped directory: {output_dir}")

print("All directories processed and zipped!")
