import torch

class MirrorFunction:
    mirror_indices = [6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4, 5]
    flip_sign = torch.tensor(
        [ 1, -1, -1, 1, 1, -1,
          1, -1, -1, 1, 1, -1], dtype=torch.float32
    )

    def action_mean_mirror(actions: torch.Tensor) -> torch.Tensor:
        """
        mirror the actions mean by swapping left and right legs
        """
        actions_mirror = actions.clone()
        mirrored = (actions_mirror * MirrorFunction.flip_sign.to(actions.device))[:, MirrorFunction.mirror_indices]
        return mirrored

    def action_std_mirror(actions: torch.Tensor) -> torch.Tensor:
        """
        mirror the actions std by swapping left and right legs
        """
        actions_mirror = actions.clone()
        mirrored = actions_mirror[:, MirrorFunction.mirror_indices]
        return mirrored

    def observation_mirror(obs_stacked: torch.Tensor, T: int = 15) -> torch.Tensor:
        """
        obs: [N, T, 47]
        return mirrored obs: [N, T, 47]
        [0]      sin(phase)
        [1]      cos(phase)
        [2-4]    command (vel_x, vel_y, yaw_rate)
        [5-16]   q
        [17-28]  dq
        [29-40]  actions
        [41-43]  base_ang_vel
        [44-46]  projected_gravity
        """
        N, tot_dim = obs_stacked.shape
        frame_dim = 47
        assert tot_dim == T * frame_dim, "obs_stacked shape mismatch"

        obs_buf = obs_stacked.reshape(N, T, frame_dim)

        obs_mirror = obs_buf.clone()
        
        # vel_y, yaw_rate: index 3,4 
        obs_mirror[:, :, 3:5] *= -1  
        # base_ang_vel.x: index 41
        obs_mirror[:, :, 41] *= -1
        # base_ang_vel.z: index 43
        obs_mirror[:, :, 43] *= -1
        # projected_gravity.y: index 45
        obs_mirror[:, :, 45] *= -1

        flip = MirrorFunction.flip_sign.to(obs_mirror.device)

        # q: index 5~16
        obs_mirror[:, :, 5:17] = obs_mirror[:, :, [5 + i for i in MirrorFunction.mirror_indices]]* flip
        # dq: index 17~28
        obs_mirror[:, :, 17:29] = obs_mirror[:, :, [17 + i for i in MirrorFunction.mirror_indices]]* flip
        # actions: index 29~40
        obs_mirror[:, :, 29:41] = obs_mirror[:, :, [29 + i for i in MirrorFunction.mirror_indices]]* flip

        return obs_mirror.reshape(N, tot_dim)