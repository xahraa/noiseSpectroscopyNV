import torch
import configurations
import funPytorch as fun
import os

# Load the configuration
conf = configurations.configN1

# Set the device to GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

try:
    # Load the model and optimizer
    model, optim, start_epoch, best_valid_metric = fun.loadModel(conf, device)
    
    # Print the loaded model and its details
    print("Loaded model:", model)
    print("Optimizer state:", optim)
    print("Starting epoch:", start_epoch)
    print("Best validation metric:", best_valid_metric)
except FileNotFoundError as e:
    print(e)

# Save the model function
def save_model(model, optim, conf, epoch, best_valid_metric):
    # Define the path where the model will be saved
    save_path = os.path.join("files", conf.path, "models")
    
    # Create the directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Define the full path for the checkpoint file
    checkpoint_path = os.path.join(save_path, "model_checkpoint.pt")
    
    # Save the model and optimizer state dictionaries along with the current epoch and best validation metric
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'epoch': epoch,
        'best_valid_metric': best_valid_metric,
    }, checkpoint_path)
    
    print(f"Model saved to {checkpoint_path}")

# Example usage
save_model(model, optim, conf, start_epoch, best_valid_metric)
