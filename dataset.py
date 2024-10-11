import torch
from torch.utils.data import Dataset

class GameDataset(Dataset):
    def __init__(self, examples):
        """
        Initializes the dataset with a list of examples.
        
        Arguments:
        - examples: List of tuples, where each tuple is (state, pi, v).
          state: Tensor of shape (1, sz, sz) for a sz x sz board state.
          pi: Tensor of shape (num_actions,) representing the target policy.
          v: Scalar tensor representing the target value.
        """
        self.examples = examples

    def __len__(self):
        """Returns the total number of examples."""
        return len(self.examples)

    def __getitem__(self, idx):
        """Retrieves the example at the given index."""
        state, pi, v = self.examples[idx]
        # Add a channel dimension to state
        expanded_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        # Clone policy tensor
        pi = pi.clone().detach().requires_grad_(True)
        # Convert value to tensor
        v = torch.tensor(v, dtype=torch.float32)
        return expanded_state, pi, v