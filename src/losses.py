import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


class NegativeSoftPlusLoss(_Loss):
    """
    Negative softplus loss.
    """
    def __init__(self) -> None:
        super(NegativeSoftPlusLoss, self).__init__()

    def forward(self, pos_scores: torch.Tensor,
                neg_scores: torch.Tensor = None) -> torch.Tensor:
        if neg_scores is None:
            neg_scores = torch.empty(0, dtype=pos_scores.dtype, device=pos_scores.device)
        input_scores = torch.cat((pos_scores * -1, neg_scores))
        return torch.mean(F.softplus(input_scores))
