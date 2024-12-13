import bpy
from utils import *

def main():

    FBX_PATH = "../meshes/MaleCharBaseMesh.fbx"
    DATA_PATH = "../data/network_solution_2_Amel_Melih_202312191406_Depart_compet_Crawl.tsv"
    BONE_CONNECTIONS = [(0,1), (1,2), (2,3), (3,4), (1,5), (5,6), (6,7),
                       (1,8), (8,9), (9,10), (1,11), (11,12), (12,13)]

    clear_scene()

    mesh_object = import_swimmer_mesh(FBX_PATH)
    motion_data = load_motion_data(DATA_PATH)


    armature, bones = create_armature()
    bind_mesh_to_armature(mesh_object, armature)

    animate_armature(motion_data, BONE_CONNECTIONS)

if __name__ == "__main__":
    main()