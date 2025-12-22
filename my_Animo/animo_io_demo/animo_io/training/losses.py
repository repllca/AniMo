from __future__ import annotations
import torch


def quat_unit_loss(q: torch.Tensor) -> torch.Tensor:
    """
    q: (B,T,J,4)
    || ||q|| - 1 || の平均
    """
    n = torch.linalg.norm(q, dim=-1)  # (B,T,J)
    return torch.mean(torch.abs(n - 1.0))


def smoothness_loss(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B,T,dim)
    速度の滑らかさ（2階差分）
    """
    if x.shape[1] < 3:
        return x.new_tensor(0.0)
    v = x[:, 1:] - x[:, :-1]          # (B,T-1,dim)
    a = v[:, 1:] - v[:, :-1]          # (B,T-2,dim)
    return torch.mean(a * a)


def mse_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - gt) ** 2)


def total_minimal_loss(pred_dict: dict[str, torch.Tensor], gt_dict: dict[str, torch.Tensor] | None = None) -> dict:
    """
    pred_dict:
      - root_translation: (B,T,3)
      - joint_quat: (B,T,J,4)
      - foot_contacts: (B,T,F)
    gt_dict: optional
      同じキーで教師ありにできる
    """
    losses = {}
    losses["quat_unit"] = quat_unit_loss(pred_dict["joint_quat"])
    losses["root_smooth"] = smoothness_loss(pred_dict["root_translation"])

    if gt_dict is not None:
        losses["root_mse"] = mse_loss(pred_dict["root_translation"], gt_dict["root_translation"])
        losses["quat_mse"] = mse_loss(pred_dict["joint_quat"], gt_dict["joint_quat"])

    losses["total"] = sum(losses.values())
    return losses
