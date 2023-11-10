import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp
from isaacgymenvs.tasks.base.vec_task import VecTask


class UR10Slide(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]


        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

    # Create environments and actors
    def create_sim(self):
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        ur10_asset_file = "urdf/ur_description/urdf/ur10_robot.urdf"

        # if "asset" in self.cfg["env"]:
        #     asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)),
        #                               self.cfg["env"]["asset"].get("assetRoot", asset_root))
        #     ur10_asset_file = self.cfg["env"]["asset"].get("assetFileName", ur10_asset_file)

        # load ur10 asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        asset_options.use_mesh_materials = True
        ur10_asset = self.gym.load_asset(self.sim, asset_root, ur10_asset_file, asset_options)

        # Create table asset
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # Create table stand asset
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.num_ur10_bodies = self.gym.get_asset_rigid_body_count(ur10_asset)
        self.num_ur10_dofs = self.gym.get_asset_dof_count(ur10_asset)

    def init_data(self):
        # Acquire tensor descriptors
        root_states_desc = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_states_desc = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_states_desc = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # PyTorch interop & Different views
        self.root_state = gymtorch.wrap_tensor(root_states_desc).view(self.num_envs, -1, 13)
        self.dof_state = gymtorch.wrap_tensor(dof_states_desc).view(self.num_envs, -1, 2)
        self.rigid_body_state = gymtorch.wrap_tensor(rigid_body_states_desc).view(self.num_envs, -1, 13)
        # Different slices
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]

    def _update_states(self):
        pass

    def _refresh(self):
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # Refresh states
        self._update_states()

    # Apply actions
    def pre_physics_step(self, actions: torch.Tensor):
        pass

    # Compute observations, rewards, and resets
    def post_physics_step(self):
        pass
