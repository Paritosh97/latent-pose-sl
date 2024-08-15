import torch
from torch import nn
from vposer import VPoser
from skeleton import Skeleton

class VPoserPredictor(nn.Module):

    def __init__(self, skeleton_path, vposer_path):
        super(VPoserPredictor, self).__init__()

        # Check if CUDA is available and set the device accordingly
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.ske = Skeleton(skeleton_path=skeleton_path)
        self.vposer = VPoser(512, 32, (1, 59, 3))

        # Load the model state dict with the appropriate device
        self.vposer.load_state_dict(torch.load(vposer_path, map_location=self.device))
        self.vposer.to(self.device)  # Move the model to the device (GPU or CPU)
        self.vposer.eval()

        for p in self.vposer.parameters():
            p.requires_grad = False

        # Initialize pose_embedding and global_trans on the correct device
        pose_embedding = torch.zeros(1, 32).to(self.device)
        self.register_buffer('pose_embedding', pose_embedding.type(torch.float32))

        global_trans = torch.zeros(1, 1, 3).to(self.device)
        self.register_buffer('global_trans', global_trans.type(torch.float32))

    def forward(self):
        pose, trans = self.get_pose()
        joints = self.ske(pose, trans)
        return joints

    def get_pose(self):
        body_pose = self.vposer.decode(self.pose_embedding, output_type='aa').view(1, 31, 3)
        return body_pose, self.global_trans
