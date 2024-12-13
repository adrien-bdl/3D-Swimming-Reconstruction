import bpy
from utils import *

def main():
    FBX_PATH = "../meshes/MaleCharBaseMesh.fbx"
    KEYPOINTS_PATH = "../data/our_swim_data.dat"
    BONE_CONNECTIONS = [
        (-2,-1), (-1,1), (1,3), (3,5), (-1,0), (0,2), (2,4),
        (-1,6), (6,8), (8,10), (-1,7), (7,9), (9,11)
    ]

    clear_scene()

    mesh_object = import_swimmer_mesh(FBX_PATH)
    motion_data = read_keypoints(KEYPOINTS_PATH)

    armature = create_base_armature()
    bones, _ = create_swimmer_bones(armature)
    
    set_bone_hierarchy(bones)

    bind_mesh_to_armature(mesh_object, armature)

    animate_swimmer_armature(motion_data, BONE_CONNECTIONS)


if __name__ == "__main__":
    main()