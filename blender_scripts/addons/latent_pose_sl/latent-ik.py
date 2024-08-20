bl_info = {
    "name": "Latent IK SL",
    "author": "Paritosh Sharma <paritosh.sharmas@gmail.com>",
    "version": (0, 0, 1),
    "blender": (4, 2, 0),
    "location": "3D View > Sidebar > Latent IK",
    "description": "IK",
    "warning": "",
    "wiki_url": "https://github.com/Paritosh97/latent-pose-sl",
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
        col.separator()

        row = col.row()
        row.prop(latentik_properties, "key_head_ik_enable")
        row = col.row()
        row.prop(latentik_properties, "key_lhand_ik_enable")
        row.prop(latentik_properties, "key_rhand_ik_enable")
        row = col.row()
        row.prop(latentik_properties, "key_hip_ik_enable")
        layout.separator()
        row = col.row()
        row.operator("anim.latentik_get_pose", icon="KEY_HLT")


class ANIM_OT_latentik_get_pose(Operator):
    bl_label = "Get Pose (Ctrl+P)"
    bl_idname = "anim.latentik_get_pose"
    bl_description = "Get Pose (Ctrl+P)"
    bl_options = {'REGISTER', 'UNDO'}

    bones = [
        "mixamorig:Spine1.001",
        "mixamorig:Spine2.001",
        "mixamorig:Neck.001",
        "mixamorig:Head.001",
        "mixamorig:LeftShoulder.001",
        "mixamorig:LeftArm.001",
        "mixamorig:LeftForeArm.001",
        "mixamorig:LeftHand.001",
        "mixamorig:LeftHandThumb1.001",
        "mixamorig:LeftHandThumb2.001",
        "mixamorig:LeftHandThumb3.001",
        "mixamorig:LeftHandIndex1.001",
        "mixamorig:LeftHandIndex2.001",
        "mixamorig:LeftHandIndex3.001",
        "mixamorig:LeftHandMiddle1.001",
        "mixamorig:LeftHandMiddle2.001",
        "mixamorig:LeftHandMiddle3.001",
        "mixamorig:LeftHandRing1.001",
        "mixamorig:LeftHandRing2.001",
        "mixamorig:LeftHandRing3.001",
        "mixamorig:LeftHandPinky1.001",
        "mixamorig:LeftHandPinky2.001",
        "mixamorig:LeftHandPinky3.001",
        "mixamorig:RightShoulder.001",
        "mixamorig:RightArm.001",
        "mixamorig:RightForeArm.001",
        "mixamorig:RightHand.001",
        "mixamorig:RightHandThumb1.001",
        "mixamorig:RightHandThumb2.001",
        "mixamorig:RightHandThumb3.001",
        "mixamorig:RightHandIndex1.001",
        "mixamorig:RightHandIndex2.001",
        "mixamorig:RightHandIndex3.001",
        "mixamorig:RightHandMiddle1.001",
        "mixamorig:RightHandMiddle2.001",
        "mixamorig:RightHandMiddle3.001",
        "mixamorig:RightHandRing1.001",
        "mixamorig:RightHandRing2.001",
        "mixamorig:RightHandRing3.001",
        "mixamorig:RightHandPinky1.001",
        "mixamorig:RightHandPinky2.001",
        "mixamorig:RightHandPinky3.001"
    ]

    mapping = dict(zip(bones, range(0, 42)))

    def execute(op, context):
        latentik_properties = context.window_manager.latentik_properties

        joint_ids = []
        joint_pos = []

        if latentik_properties.key_hip_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_hip_ik_name].location
            joint_ids.append(op.mapping["mixamorig:Spine1.001"])  # Adjust mapping accordingly
            joint_pos.extend(pos)

        if latentik_properties.key_lhand_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_lhand_ik_name].location
            joint_ids.append(op.mapping["mixamorig:LeftHand.001"])
            joint_pos.extend(pos)

        if latentik_properties.key_rhand_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_rhand_ik_name].location
            joint_ids.append(op.mapping["mixamorig:RightHand.001"])
            joint_pos.extend(pos)
        
        if latentik_properties.key_head_ik_enable:
            pos = bpy.data.objects[latentik_properties.controller_head_ik_name].location
            joint_ids.append(op.mapping["mixamorig:Head.001"])
            joint_pos.extend(pos)

        # Process the pose directly in Blender
        result = op.calculate_pose(joint_ids, joint_pos)

        arm = bpy.data.objects[latentik_properties.armature_name]
        if arm is None:
            for ob in bpy.data.objects:
                if ob.type == "ARMATURE":
                    arm = ob
                    latentik_properties.armature_name = ob.name

        op.apply_pose(arm, result)

        return {'FINISHED'}

    def calculate_pose(self, joint_ids, joint_pos):
        # Dummy calculation logic - this should be replaced with actual IK processing logic
        # Here we just return some mock data
        rot = [[1, 0, 0, 0]] * 42  # Replace with actual rotations
        trans = [0, 0, 0]  # Replace with actual translation
        return {"pose": rot, "trans": trans}

    def apply_pose(self, arm, pose_param):
        rot = pose_param["pose"]
        trans = pose_param["trans"]
        arm.location = trans
        for pbone in arm.pose.bones:
            this_bone_id = self.mapping.get(pbone.name)
            if this_bone_id is not None:
                rotation = rot[this_bone_id]
                pbone.rotation_mode = "AXIS_ANGLE"
                pbone.rotation_axis_angle = rotation
                pbone.scale = [1, 1, 1]


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
    kmi = km.keymap_items.new("anim.latentik_get_pose", type="P", ctrl=True, value="PRESS")
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
