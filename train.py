import torch
import torch.nn.functional as F

def compute_loss(pred_policy, pred_value, target_policy, target_value, mse_weight = 0.01):
    """
    Computes the loss for a batch of game examples.
    
    Arguments:
    - pred_policy: Tensor of shape [batch_size, num_actions], predicted policy p.
    - pred_value: Tensor of shape [batch_size, 1], predicted value z.
    - target_policy: Tensor of shape [batch_size, num_actions], target improved policy pi.
    - target_value: Tensor of shape [batch_size, 1], target value estimate v.
    - mse_weight: Value to weight the MSE loss by. Small to avoid overfitting to value.
    
    Returns:
    - total_loss: The combined value loss and policy loss for the batch.
    """
    # Value loss: Mean Squared Error (MSE) between predicted value and target value
    value_loss = F.mse_loss(pred_value, target_value)
    
    # Policy loss: Cross-entropy (negative log likelihood) between target policy and predicted policy
    policy_loss = -(target_policy * torch.log(pred_policy + 1e-10)).sum(dim=1).mean()
    
    # Total loss is the sum of value loss and policy loss
    total_loss = mse_weight*value_loss + policy_loss
    
    return total_loss

def train(model, optimizer, epochs, dataloader):
    """
    Trains the model using the Adam optimizer and the loss function.

    Arguments:
    - model: The neural network (CNN).
    - optimizer: The optimizer (Adam).
    - epochs: Number of training epochs.
    - dataloader: A DataLoader that provides batches of (state, target_policy, target_value).
    """
    model.train()  # Set the model to training mode
    
    for epoch in range(epochs):
        epoch_loss = 0.0

        for batch in dataloader:
            states, target_policies, target_values = batch  # Dataloader returns tuples of (state, pi, v)

            # Forward pass: get the predicted policy and value
            pred_policy, pred_value = model(states)

            # Compute the loss
            loss = compute_loss(pred_policy, pred_value, target_policies, target_values)

            # Backward pass: compute the gradients
            optimizer.zero_grad()  # Zero the parameter gradients
            loss.backward()  # Backpropagate the loss

            # Perform one step of optimization (parameter update)
            optimizer.step()

            # Accumulate the loss for reporting
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')