import torch
from torch import nn
from .vposer import VPoser
from .skeleton import Skeleton

class VPoserPredictor(nn.Module):

    def __init__(self, skeleton_path, vposer_path):
        super(VPoserPredictor, self).__init__()

        # Check if CUDA is available and set the device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the skeleton model (used to reconstruct joint positions)
        self.ske = Skeleton(skeleton_path=skeleton_path)

        # Initialize VPoser with the appropriate latent dimension and joint size
        latentD = 32  # Set to the correct latent dimension used during training
        self.vposer = VPoser(num_neurons=512, latentD=latentD, data_shape=(1, 42, 3))  # 42 joints
        self.vposer.load_state_dict(torch.load(vposer_path, map_location=self.device))  # Load the trained VPoser model
        self.vposer.to(self.device)  # Move the VPoser model to the appropriate device
        self.vposer.eval()

        # Freeze the VPoser model parameters (as done in training)
        for p in self.vposer.parameters():
            p.requires_grad = False

        # Register pose_embedding and global_trans buffers directly
        self.register_buffer('pose_embedding', torch.zeros(1, latentD, dtype=torch.float32).to(self.device))
        self.register_buffer('global_trans', torch.zeros(1, 1, 3, dtype=torch.float32).to(self.device))

    def forward(self):
        # Decode the latent pose embedding into the 42-joint pose
        pose, trans = self.get_pose()
        joints = self.ske(pose, trans)
        return joints

    def get_pose(self):
        # Decode latent pose embedding into axis-angle representation (42 joints for body, 1 global)
        body_pose = self.vposer.decode(self.pose_embedding, output_type='aa').view(1, 42, 3)  # Output shape from training
        return body_pose, self.global_trans
