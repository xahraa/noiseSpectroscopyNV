import torch
import configurations
import funPytorch as fun
import loaders
import os
import numpy as np

# Load the configuration
conf = configurations.configN1

# Set the device to GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model and optimizer
model, optim, start_epoch, best_valid_metric = fun.loadModel(conf, device)

# Load the data
dataloaders, datasets = fun.processData(conf)

# Function to make predictions
def make_predictions(model, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    true_values = []
    with torch.no_grad():  # Disable gradient calculation
        for batch in dataloader:
            inputs = {k: batch[k].to(device) for k in batch}
            outputs = model(inputs)
            predictions.extend(outputs['y'].cpu().numpy())
            true_values.extend(inputs['y'].cpu().numpy())
    return np.array(predictions), np.array(true_values)

# Make predictions on the test set
test_loader = dataloaders['test']
predictions, true_values = make_predictions(model, test_loader, device)

# Save predictions to a file
predictions_file = os.path.join("files", conf.path, "predictions", "test_predictions.npz")
os.makedirs(os.path.dirname(predictions_file), exist_ok=True)
np.savez_compressed(predictions_file, predictions=predictions, true_values=true_values)
print(f"Predictions saved to {predictions_file}")
