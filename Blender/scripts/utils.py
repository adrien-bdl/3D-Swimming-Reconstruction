import bpy
import pandas as pd
import mathutils
from math import *
import numpy as np

def clear_scene():
    if bpy.context.mode != 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')
    
    # Clear animations
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='ARMATURE')
    bpy.ops.object.delete()
    
    # Clear objects
    bpy.ops.object.select_all(action='DESELECT')
    bpy.ops.object.select_by_type(type='MESH')
    bpy.ops.object.delete()

def import_swimmer_mesh(fbx_path):
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    
    # Remove extra mesh if present
    if "Char_lp" in bpy.data.objects:
        bpy.ops.object.select_all(action='DESELECT')
        bpy.data.objects['Char_lp'].select_set(True)
        bpy.ops.object.delete()
    
    mesh_object = bpy.data.objects['Char_Mlp']
    mesh_object.location = (0, 0, 0)
    mesh_object.rotation_euler[1] = radians(-90)
    
    return mesh_object

def load_motion_data(data_path):
    column_names = ['t','CM_X','CM_Y'] + [f'{i}_{coord}' for i in range(14) for coord in ['X','Y','P']] + ['z_plane']
    return pd.read_csv(data_path, delim_whitespace=True, comment='#', names=column_names)

def create_armature():
    bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0))
    armature = bpy.context.object
    
    # Clear default bone
    bpy.ops.object.editmode_toggle()
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    bpy.ops.object.editmode_toggle()
    
    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    bones = create_bones(armature)
    set_bone_hierarchy(bones)
    
    return armature, bones

def create_base_armature():
    """Create a basic armature without bones."""
    bpy.ops.object.armature_add(enter_editmode=True, align='WORLD', location=(0, 0, 0))
    armature = bpy.context.object
    
    # Clear default bone
    bpy.ops.object.editmode_toggle()
    bpy.ops.armature.select_all(action='SELECT')
    bpy.ops.armature.delete()
    bpy.ops.object.editmode_toggle()
    
    # Switch to edit mode
    bpy.ops.object.mode_set(mode='EDIT')
    
    return armature

def create_bones(armature):
    bones = []
    bone_data = [
        ("Bone0to1", (-1.9, 0, 0), (-1.7, 0, 0)),
        ("Bone1to2", (-1.7, 0, 0), (-1.58, 0, 0.23)),
        ("Bone2to3", (-1.58, 0, 0.23), (-1.36, 0, 0.33)),
        ("Bone3to4", (-1.36, 0, 0.33), (-1.1, 0, 0.47)),
        ("Bone1to5", (-1.7, 0, 0), (-1.58, 0, -0.23)),
        ("Bone5to6", (-1.58, 0, -0.23), (-1.36, 0, -0.33)),
        ("Bone6to7", (-1.36, 0, -0.33), (-1.1, 0, -0.47)),
        ("Bone1to8", (-1.7, 0, 0), (-1.1, 0, 0.08)),
        ("Bone8to9", (-1.1, 0, 0.08), (-0.6, 0, 0.09)),
        ("Bone9to10", (-0.6, 0, 0.09), (0, 0, 0.18)),
        ("Bone1to11", (-1.7, 0, 0), (-1.1, 0, -0.08)),
        ("Bone11to12", (-1.1, 0, -0.08), (-0.6, 0, -0.09)),
        ("Bone12to13", (-0.6, 0, -0.09), (0, 0, -0.18))
    ]
    
    for name, head, tail in bone_data:
        bone = armature.data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        bones.append(bone)
    
    return bones

def set_bone_hierarchy(bones):
    hierarchy = [
        (1, 0), (2, 1), (3, 2),
        (4, 0), (5, 4), (6, 5),
        (7, 0), (8, 7), (9, 8),
        (10, 0), (11, 10), (12, 11)
    ]
    
    for child, parent in hierarchy:
        bones[child].parent = bones[parent]

def bind_mesh_to_armature(mesh_object, armature):
    bpy.ops.object.mode_set(mode='OBJECT')
    
    bpy.ops.object.select_all(action='DESELECT')
    mesh_object.select_set(True)
    armature.select_set(True)
    
    bpy.context.view_layer.objects.active = armature
    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

def animate_armature(df, bone_connections):
    """Create animation keyframes based on motion data."""
    bpy.ops.object.mode_set(mode='POSE')
    
    coord_prof = [0, 0, 0.23, 0.33, 0.47, -0.23, -0.33, -0.47, 0.08, 0.09, 0.14, -0.08, -0.09, -0.14]
    
    for index, row in df.iterrows():
        if index % 2 == 0:
            continue
            
        bpy.context.scene.frame_set(index//2-10)
        directions = [np.array([0, 0, 0], float) for _ in range(13)]
        directions[0] = np.array([0.2, 0, 0])
        
        animate_frame(row, bone_connections, directions, coord_prof)

def animate_frame(row, bone_connections, directions, coord_prof):
    """Animate a single frame for all bones."""
    for k, bone in enumerate(bpy.context.object.pose.bones):
        if k == 0:
            continue
            
        i, j = bone_connections[k]
        head_loc = (row[f'{i}_X'], row[f'{i}_Y'], coord_prof[i])
        tail_loc = (row[f'{j}_X'], row[f'{j}_Y'], coord_prof[j])
        
        direction_abs = calculate_direction(head_loc, tail_loc)
        directions[k] = direction_abs
        
        # Find parent direction
        for m, (o, p) in enumerate(bone_connections):
            if p == i:
                direction_parent = directions[m]
                break
        
        rotation = calculate_rotation(direction_parent, direction_abs)
        apply_rotation(bone, rotation)

def calculate_direction(head_loc, tail_loc):
    return np.array([
        tail_loc[0] - head_loc[0],
        tail_loc[1] - head_loc[1],
        tail_loc[2] - head_loc[2]
    ], float)

def calculate_rotation(direction_parent, direction_abs):
    """Calculate rotation quaternion from parent and absolute directions."""
    cross_product = np.cross(direction_parent, direction_abs)
    axis = cross_product / np.linalg.norm(cross_product)
    
    parent_norm = np.linalg.norm(direction_parent)
    abs_norm = np.linalg.norm(direction_abs)
    
    sin_angle = np.linalg.norm(cross_product) / (parent_norm * abs_norm)
    cos_angle = np.dot(direction_parent, direction_abs) / (parent_norm * abs_norm)
    
    angle = 2 * pi - acos(cos_angle) if sin_angle < 0 else acos(cos_angle)
    
    return mathutils.Quaternion(axis, angle)

def apply_rotation(bone, rotation):
    bone.rotation_quaternion = rotation
    bone.keyframe_insert(data_path='location', index=-1)
    bone.keyframe_insert(data_path='rotation_quaternion', index=-1)


def read_keypoints(filename):
    pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])
    
    with open(filename, 'r') as fin:
        kpts = []
        while True:
            line = fin.readline()
            if line == '':
                break
                
            line = [float(s) for s in line.split()]
            line = np.reshape(line, (len(pose_keypoints), -1))
            kpts.append(line)
            
    return np.array(kpts)

def create_swimmer_bones(armature):
    bones = []
    length_bones = []
    
    bone_data = [
        ("Bone-2to-1", (-1.9, 0, 0), (-1.7, 0, 0)),
        ("Bone-1to0", (-1.7, 0, 0), (-1.58, 0, 0.23)),
        ("Bone0to2", (-1.58, 0, 0.23), (-1.36, 0, 0.33)),
        ("Bone2to4", (-1.36, 0, 0.33), (-1.1, 0, 0.47)),
        ("Bone-1to1", (-1.7, 0, 0), (-1.58, 0, -0.23)),
        ("Bone1to3", (-1.58, 0, -0.23), (-1.36, 0, -0.33)),
        ("Bone3to5", (-1.36, 0, -0.33), (-1.1, 0, -0.47)),
        ("Bone-1to6", (-1.7, 0, 0), (-1.1, 0, -0.08)),
        ("Bone6to8", (-1.1, 0, -0.08), (-0.6, 0, -0.09)),
        ("Bone8to10", (-0.6, 0, -0.09), (0, 0, -0.18)),
        ("Bone-1to7", (-1.7, 0, 0), (-1.1, 0, 0.08)),
        ("Bone7to9", (-1.1, 0, 0.08), (-0.6, 0, 0.09)),
        ("Bone9to11", (-0.6, 0, 0.09), (0, 0, 0.18))
    ]
    
    for name, head, tail in bone_data:
        bone = armature.data.edit_bones.new(name)
        bone.head = head
        bone.tail = tail
        bones.append(bone)
        length_bones.append(dist(bone.head, bone.tail))
    
    return bones, length_bones

def animate_swimmer_armature(p3ds, bone_connections):
    bpy.ops.object.mode_set(mode='POSE')
    
    for index, kpts3d in enumerate(p3ds):
        bpy.context.scene.frame_set(index)
        directions = [np.array([0, 0, 0], float) for _ in range(13)]
        directions[0] = np.array([0.2, 0, 0])
        
        animate_swimmer_frame(kpts3d, bone_connections, directions)

def animate_swimmer_frame(kpts3d, bone_connections, directions):
    for k, bone in enumerate(bpy.context.object.pose.bones):
        if k == 0:
            continue
            
        i, j = bone_connections[k]
        
        # Handle special case for middle point between keypoints
        if i == -1:
            head_loc = (
                (kpts3d[1, 0] + kpts3d[0, 0])/2,
                (kpts3d[1, 1] + kpts3d[0, 1])/2,
                (kpts3d[1, 2] + kpts3d[0, 2])/2
            )
        else:
            head_loc = (kpts3d[i, 0], kpts3d[i, 1], kpts3d[i, 2])
            
        tail_loc = (kpts3d[j, 0], kpts3d[j, 1], kpts3d[j, 2])
        
        direction_abs = calculate_direction(head_loc, tail_loc)
        directions[k] = direction_abs
        
        # Find parent direction
        for m, (o, p) in enumerate(bone_connections):
            if p == i:
                direction_parent = directions[m]
                break
        
        rotation = calculate_rotation(direction_parent, direction_abs)
        apply_rotation(bone, rotation)