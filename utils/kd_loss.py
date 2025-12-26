import torch
import torch.nn.functional as F


def kd_cosine_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """
    Cosine embedding distillation.
    Both input tensors: [B, D]
    """
    s = F.normalize(student_emb, dim=1)
    t = F.normalize(teacher_emb, dim=1)
    return (1.0 - (s * t).sum(dim=1)).mean()


def kd_feature_loss(student_fm: torch.Tensor, teacher_fm: torch.Tensor) -> torch.Tensor:
    """
    Feature map distillation. Expect same [B, C, H, W].
    If not same, handle alignment outside (1x1 adapter + interpolate).
    """
    s = F.normalize(student_fm, dim=1)
    t = F.normalize(teacher_fm, dim=1)
    return F.mse_loss(s, t)


def kd_relation_loss(student_emb: torch.Tensor, teacher_emb: torch.Tensor) -> torch.Tensor:
    """
    Relational KD: match pairwise similarity matrix in a batch.
    """
    s = F.normalize(student_emb, dim=1)
    t = F.normalize(teacher_emb, dim=1)
    sim_s = s @ s.t()
    sim_t = t @ t.t()
    return F.mse_loss(sim_s, sim_t)

def total_loss(
    student_emb: torch.Tensor,
    teacher_emb: torch.Tensor,
    student_fm: torch.Tensor = None,
    teacher_fm: torch.Tensor = None,
    w_emb: float = 1.0,
    w_fm: float = 0.1,
    w_rel: float = 0.5,
) -> torch.Tensor:
    # 1) embedding KD
    loss = w_emb * kd_cosine_loss(student_emb, teacher_emb)

    # 2) feature map KD（如果传了就算）
    if (student_fm is not None) and (teacher_fm is not None):
        loss = loss + w_fm * kd_feature_loss(student_fm, teacher_fm)

    # 3) relation KD
    loss = loss + w_rel * kd_relation_loss(student_emb, teacher_emb)

    return loss