import numpy as np
import torch
import random
import os
import json
import pandas as pd
import nbformat
from scipy.interpolate import griddata

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # Ensure GPU determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed = '+str(seed))

def count_learnable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def linear_interpolation_1D(x, xp, fp):
    # Detach tensors and move to CPU
    x_cpu = x.detach().cpu().numpy()
    xp_cpu = xp.detach().cpu().numpy()
    fp_cpu = fp.detach().cpu().numpy()
    
    # Perform linear interpolation using np.interp
    y_cpu = np.interp(x_cpu, xp_cpu, fp_cpu)
    
    # Convert the result back to torch tensor and move to device
    y = torch.tensor(y_cpu, device=x.device)
    return y

def linear_interpolation_2D(x, y, xp, yp, fp):
    """
    Perform bilinear interpolation for a 2D field with output as (M, 1).

    Args:
        x (torch.Tensor): x-coordinates where the field is evaluated (1D tensor of size M).
        y (torch.Tensor): y-coordinates where the field is evaluated (1D tensor of size M).
        xp (torch.Tensor): x-coordinates of the known field values.
        yp (torch.Tensor): y-coordinates of the known field values.
        fp (torch.Tensor): 2D tensor of field values at (xp, yp).

    Returns:
        torch.Tensor: Interpolated field as a column vector of size (M, 1).
    """
    # Detach tensors and move to CPU
    x_cpu = x.detach().cpu().numpy()
    y_cpu = y.detach().cpu().numpy()
    xp_cpu = xp.detach().cpu().numpy().flatten()
    yp_cpu = yp.detach().cpu().numpy().flatten()
    fp_cpu = fp.detach().cpu().numpy().flatten()

    # Combine xp and yp to form the grid of known points
    points = np.column_stack((xp_cpu, yp_cpu))  # (N, 2) array

    # Points to evaluate
    query_points = np.column_stack((x_cpu, y_cpu))  # (M, 2) array

    # Perform interpolation using griddata
    interpolated_values = griddata(points, fp_cpu, query_points, method='linear')

    # Convert back to PyTorch tensor and reshape to (M, 1)
    interpolated_tensor = torch.tensor(interpolated_values, device=x.device, dtype=x.dtype).view(-1, 1)

    return interpolated_tensor

def performance_metrics(mse_test, r2score_test, relerror_test, training_time, runtime_per_iter, resultdir, save):
    print("Mean Squared Error Test:\n" + str(mse_test))
    print("R2 score Test:\n" + str(r2score_test))
    print("Rel. L2 Error Test:\n" + str(relerror_test))
    print("Training Time (in sec):\n" + str(training_time))  # Time taken for network training
    print(f"Runtime per Iteration (in sec/iter):\n{(runtime_per_iter):.3f}")

    performance_metrics = {
        "Mean Squared Error Test": mse_test,
        "R2 score Test": r2score_test,
        "Rel. L2 Error Test": relerror_test,
        "Training Time (in sec)": training_time,
        "Runtime per Iteration (in sec/iter)": runtime_per_iter
    }
    
    if save == True:
        # Save the performance_metrics to a JSON file
        with open(os.path.join(resultdir, "performance_metrics.json"), "w") as file:
            json.dump(performance_metrics, file, indent=4)


def process_jsons_in_folder(base_folder, seed_values, n_used_values=None):
    """
    Process jsons in a folder to extract performance metrics.
    """
    # Initialize an empty list to store the data
    data = []
    
    # Check if n_used_values is provided
    if n_used_values is None:
        n_used_values = [None]

    # Traverse the directory structure and read JSON files
    for n_used in n_used_values:
        for seed in seed_values:
            if n_used is not None:
                folder_path = os.path.join(base_folder, f'seed={seed}_n_used={n_used}')
            else:
                folder_path = os.path.join(base_folder, f'seed={seed}')

            file_path = os.path.join(folder_path, 'performance_metrics.json')
            
            # Check if the file exists
            if os.path.exists(file_path):
                # Read the JSON file
                with open(file_path, 'r') as f:
                    metrics = json.load(f)

                # Extract MSE_test
                mse_test = metrics.get('Mean Squared Error Test')
                training_time = metrics.get('Training Time (in sec)')
                runtime_per_iter = metrics.get('Runtime per Iteration (in sec/iter)')

                # Store the data in a dictionary and append to the list
                data.append({
                    'seed': seed,
                    'n_used': n_used,
                    'Mean Squared Error Test': mse_test,
                    'Training Time (in sec)': training_time,
                    'Runtime per Iteration (in sec/iter)': runtime_per_iter
                })

    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)

    return df


def process_notebooks_in_folder(base_folder, seed_values, n_used_values=None):
    """
    Process notebooks in a folder to extract performance metrics.
    """
    data = []
    
    # Default n_used values if not provided
    if n_used_values is None:
        n_used_values = [None]

    # Iterate through seed and n_used combinations
    for n_used in n_used_values:
        for seed in seed_values:
            if n_used is not None:
                folder_path = os.path.join(base_folder, f'seed={seed}_n_used={n_used}')
                file_path = os.path.join(folder_path, f'output_seed={seed}_n_used={n_used}.ipynb')
            else:
                folder_path = os.path.join(base_folder, f'seed={seed}')
                file_path = os.path.join(folder_path, f'output_seed={seed}.ipynb')
            
            # Check if the notebook exists
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    notebook = nbformat.read(f, as_version=4)

                # Extract from second-last cell (or adjust as necessary)
                last_cell = notebook.cells[-3]
                
                # Initialize metrics
                mse, r2_score, rel_l2_error, training_time, runtime_per_iter = [None] * 5
                
                # Extract outputs
                if 'outputs' in last_cell and last_cell['outputs']:
                    for output in last_cell['outputs']:
                        if output.output_type == 'stream':
                            lines = output['text'].splitlines()

                            # Parse output lines
                            for i, line in enumerate(lines):
                                if "Mean Squared Error Test" in line:
                                    mse = float(lines[i + 1].strip())
                                elif "R2 score Test" in line:
                                    r2_score = float(lines[i + 1].strip())
                                elif "Rel. L2 Error Test" in line:
                                    rel_l2_error = float(lines[i + 1].strip())
                                elif "Training Time (in sec)" in line:
                                    training_time = float(lines[i + 1].strip())
                                elif "Runtime per Iteration (in sec/iter)" in line:
                                    runtime_per_iter = float(lines[i + 1].strip())

                # Append metrics to data list
                data.append({
                    'seed': seed,
                    'n_used': n_used,
                    'Mean Squared Error Test': mse,
                    'R2 score Test': r2_score,
                    'Rel. L2 Error Test': rel_l2_error,
                    'Training Time (in sec)': training_time,
                    'Runtime per Iteration (in sec/iter)': runtime_per_iter
                })
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    return df