"""Skip frozen output branches that do not feed event logits during SHAP."""
import torch
class UnusedTrajectoryOutput(torch.nn.Module):
    def __init__(self,outputs):super().__init__();self.outputs=outputs
    def forward(self,x):return x.new_zeros((*x.shape[:-1],self.outputs))
def omit_unused_heads(model):
    # forward() computes encoder -> outcome_head + family_head first. All
    # removed branches consume encoded states but never feed either logits.
    # This function changes no retained weight and no model/data core file.
    model.value_head=None
    model.time_head=None
    model.time_distribution_head=None
    model.dual_hazard_head=None
    model.tail_distribution_head=None
    model.trajectory_head=UnusedTrajectoryOutput(model.num_trajectory_horizons*model.num_outcomes)
    return model
