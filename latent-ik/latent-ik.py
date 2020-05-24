# -*- coding: utf-8 -*-
#
# Author: maajor <info@ma-yidong.com>
# Date : 2020-05-23
# 
# Blender plugin to fetch predicted pose from server and apply on armature

bl_info = {
    "name": "Latent IK",
    "author": "Ma Yi Dong <info@ma-yidong.com>",
    "version": (0, 0, 1),
    "blender": (2, 80, 0),
    "location": "3D View > Sidebar > Latent IK",
    "description": "IK",
    "warning": "",
    "wiki_url": "https://github.com/maajor/latent-pose",
    "category": "Animation",
}

import bpy
from bpy.types import (
        Operator,
        Panel,
        AddonPreferences,
        )
from bpy.props import (
        BoolProperty,
        StringProperty,
        )
from bpy.app.handlers import persistent

import requests
import numpy as np


# Property Definitions
class LatentIKProperty(bpy.types.PropertyGroup):
    key_hip_ik_enable: BoolProperty(
        name="Hip IK Enable",
        description="Hip IK Enable",
        default=True
    )
    key_lhand_ik_enable: BoolProperty(
        name="LeftHand IK Enable",
        description="LeftHand IK Enable",
        default=True
    )
    key_rhand_ik_enable: BoolProperty(
        name="RightHand IK Enable",
        description="RightHand IK Enable",
        default=True
    )
    key_head_ik_enable: BoolProperty(
        name="Head IK Enable",
        description="Head IK Enable",
        default=True
    )
    key_lfoot_ik_enable: BoolProperty(
        name="LeftFoot IK Enable",
        description="LeftFoot IK Enable",
        default=True
    )
    key_rfoot_ik_enable: BoolProperty(
        name="RightFoot IK Enable",
        description="RightFoot IK Enable",
        default=True
    )
    server_address: StringProperty(
        name="Server Address",
        description="Server Address",
        default="127.0.0.1:1028"
    )
    controller_head_ik_name: StringProperty(
        name="Head IK Controller",
        description="Head IK Controller",
        default="HeadController"
    )
    controller_lhand_ik_name: StringProperty(
        name="LeftHand IK Controller",
        description="LeftHand IK Controller",
        default="LeftHandController"
    )
    controller_rhand_ik_name: StringProperty(
        name="RightHand IK Controller",
        description="RightHand IK Controller",
        default="RightHandController"
    )
    controller_hip_ik_name: StringProperty(
        name="Hip IK Controller",
        description="Hip IK Controller",
        default="HipController"
    )
    controller_lfoot_ik_name: StringProperty(
        name="LeftFoot IK Controller",
        description="LeftFoot IK Controller",
        default="LeftFootController"
    )
    controller_rfoot_ik_name: StringProperty(
        name="RightFoot IK Controller",
        description="RightFoot IK Controller",
        default="RightFootController"
    )
    armature_name: StringProperty(
        name="Armature Name",
        description="Armature Name",
        default="character"
    )

# GUI (Panel)

class VIEW3D_PT_LatentIKUI(Panel):
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Animate"
    bl_label = 'Latent-IK'
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(self, context):
        return context.active_object and context.active_object.type in {'ARMATURE'}

    def draw(self, context):
        obj = context.active_object
        latentik_properties = context.window_manager.latentik_properties

        layout = self.layout
        col = layout.column(align=True)
        row = col.row()
        row.label(text='Latent IK Controller', icon="INFO")
        row = col.row()
        row.prop(latentik_properties, "server_address")
        col.separator()

        row = col.row()
        row.prop(latentik_properties, "key_head_ik_enable")
        row = col.row()
        row.prop(latentik_properties, "key_lhand_ik_enable")
        row.prop(latentik_properties, "key_rhand_ik_enable")
        row = col.row()
        row.prop(latentik_properties, "key_hip_ik_enable")
        row = col.row()
        row.prop(latentik_properties, "key_lfoot_ik_enable")
        row.prop(latentik_properties, "key_rfoot_ik_enable")
        layout.separator()
        row = col.row()
        row.operator("anim.latentik_get_pose", icon="KEY_HLT")


param_dict = {}
param_dict["joint_pos"] =  [0, 0, 0,-2, 6, 6, 1, -1, 7]
param_dict["joint_id"] = [0, 20, 16] # left hand and head
resp = requests.get("http://127.0.0.1:1028/predict",
                     params=param_dict)
result = resp.json()
print(result)

class ANIM_OT_latentik_get_pose(Operator):
    bl_label = "Get Pose (Ctrl+P)"
    bl_idname = "anim.latentik_get_pose"
    bl_description = "Get Pose (Ctrl+P)"
    bl_options = {'REGISTER', 'UNDO'}
    mapping = {'Head': 16, 'Hips': 0, 'LHipJoint': 1, 'LThumb': 23, 'LeftArm': 18, 
    'LeftFingerBase': 21, 'LeftFoot': 4, 'LeftForeArm': 19, 'LeftHand': 20, 'LeftHandIndex1': 22, 
    'LeftLeg': 3, 'LeftShoulder': 17, 'LeftToeBase': 5, 'LeftUpLeg': 2, 'LowerBack': 11, 'Neck': 14, 
    'Neck1': 15, 'RHipJoint': 6, 'RThumb': 30, 'RightArm': 25, 'RightFingerBase': 28, 'RightFoot': 9, 
    'RightForeArm': 26, 'RightHand': 27, 'RightHandIndex1': 29, 'RightLeg': 8, 'RightShoulder': 24, 
    'RightToeBase': 10, 'RightUpLeg': 7, 'Spine': 12, 'Spine1': 13}

    def execute(op, context):
        latentik_properties = context.window_manager.latentik_properties

        joint_ids = []
        joint_pos = []

        if latentik_properties.key_hip_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_hip_ik_name].location
            joint_ids.append(op.mapping["Hips"])
            joint_pos.extend(pos)

        if latentik_properties.key_lhand_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_lhand_ik_name].location
            joint_ids.append(op.mapping["LeftHand"])
            joint_pos.extend(pos)

        if latentik_properties.key_rhand_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_rhand_ik_name].location
            joint_ids.append(op.mapping["RightHand"])
            joint_pos.extend(pos)
        
        if latentik_properties.key_head_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_head_ik_name].location
            joint_ids.append(op.mapping["Head"])
            joint_pos.extend(pos)

        if latentik_properties.key_lfoot_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_lfoot_ik_name].location
            joint_ids.append(op.mapping["LeftFoot"])
            joint_pos.extend(pos)

        if latentik_properties.key_rfoot_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_rfoot_ik_name].location
            joint_ids.append(op.mapping["RightFoot"])
            joint_pos.extend(pos)

        param_dict = {}
        param_dict["joint_pos"] =  joint_pos
        param_dict["joint_id"] = joint_ids

        result = op.query_pose_from_server(latentik_properties.server_address, param_dict)

        arm = bpy.data.objects[latentik_properties.armature_name]
        if arm is None:
            for ob in bpy.data.objects:
                if ob.type is "ARMATURE":
                    arm = ob
                    latentik_properties.armature_name = ob.name

        op.apply_pose(arm, result)

        return {'FINISHED'}

    def query_pose_from_server(self, server_address, param_dict):
        resp = requests.get("http://{0}/predict".format(server_address),
                     params=param_dict)
        result = resp.json()
        return result

    def apply_pose(self, arm, pose_param):
        rot = pose_param["pose"]
        trans = pose_param["trans"]
        arm.location = trans
        for pbone in arm.pose.bones:
            this_bone_id = self.mapping[pbone.name]
            rotation = rot[this_bone_id]
            pbone.rotation_mode = "AXIS_ANGLE"
            pbone.rotation_axis_angle = rotation
            pbone.scale = [1,1,1]


# Add-ons Preferences Update Panel

# Define Panel classes for updating
panels = [
        VIEW3D_PT_LatentIKUI
        ]


def update_panel(self, context):
    message = "Latent IK: Updating Panel locations has failed"
    try:
        for panel in panels:
            if "bl_rna" in panel.__dict__:
                bpy.utils.unregister_class(panel)

        for panel in panels:
            panel.bl_category = context.preferences.addons[__name__].preferences.category
            bpy.utils.register_class(panel)

    except Exception as e:
        print("\n[{}]\n{}\n\nError:\n{}".format(__name__, message, e))
        pass


class LatentIKAddonPreferences(AddonPreferences):
    bl_idname = __name__

    category: StringProperty(
        name="Tab Category",
        description="Choose a name for the category of the panel",
        default="Animate",
        update=update_panel
    )

    def draw(self, context):
        layout = self.layout
        row = layout.row()
        col = row.column()

        col.label(text="Tab Category:")
        col.prop(self, "category", text="")

addon_keymaps = []

def register():
    bpy.utils.register_class(LatentIKProperty)
    bpy.types.WindowManager.latentik_properties = bpy.props.PointerProperty(type=LatentIKProperty)
    bpy.utils.register_class(VIEW3D_PT_LatentIKUI)
    bpy.utils.register_class(ANIM_OT_latentik_get_pose)
    bpy.utils.register_class(LatentIKAddonPreferences)

    global addon_keymaps
    wm = bpy.context.window_manager
    km = wm.keyconfigs.addon.keymaps.new(name='Object Mode', space_type='EMPTY')
    # Ctrl+P for shortcut
    kmi = km.keymap_items.new("anim.latentik_get_pose", type = "P", ctrl=True, value = "PRESS")
    addon_keymaps.append(km)
    update_panel(None, bpy.context)


def unregister():
    del bpy.types.WindowManager.latentik_properties
    bpy.utils.unregister_class(VIEW3D_PT_LatentIKUI)
    bpy.utils.unregister_class(ANIM_OT_latentik_get_pose)
    bpy.utils.unregister_class(LatentIKAddonPreferences)
    wm = bpy.context.window_manager
    for km in addon_keymaps:
        wm.keyconfigs.addon.keymaps.remove(km)
    # clear the list
    del addon_keymaps[:]

if __name__ == "__main__":
    register()
