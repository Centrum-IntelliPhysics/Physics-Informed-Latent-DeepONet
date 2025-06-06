import nbformat



# Function to extract runtime and memory usage from a notebook file
def extract_runtime_and_memory_from_notebook(notebook_path):
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = nbformat.read(f, as_version=4)
    
    # Get the last relevant cell (adjust if necessary)
    last_cell = notebook.cells[-3]  # Adjust based on the required cell

    runtime_per_iter = None
    memory_usage = None

    # Check if the last cell has outputs
    if 'outputs' in last_cell and last_cell['outputs']:
        for output in last_cell['outputs']:
            if output.output_type == 'stream':  # For standard output (print statements, etc.)
                output_text = output['text']
                
                # Extract the runtime and memory information
                lines = output_text.splitlines()

                # Look for "Runtime per Iteration (in sec/iter):" and extract the value
                if "Runtime per Iteration (in sec/iter):" in lines[8]:
                    runtime_per_iter = float(lines[9])  # Extract runtime in seconds
                # Look for "Memory used (in MB):" and extract the value in MB
                if "Memory used (in MB):" in lines[10]:
                    memory_usage = float(lines[11])  # Extract memory in MB

    return runtime_per_iter, memory_usage