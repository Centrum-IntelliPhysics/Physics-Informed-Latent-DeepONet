import os

# Define the values
seed_values = [0]
n_used_values = [0, 150, 300]

# Iterate over each combination
for seed in seed_values:
    for n_used in n_used_values:
        # Construct the directory path
        resultdir = os.path.join(os.getcwd(), 'results', 'b_Latent-NO', f'seed={seed}_n_used={n_used}')
        if not os.path.exists(resultdir):
            os.makedirs(resultdir)

        # Set save to True only for seed 0
        save = True if seed == 0 else False
        
        # Construct the command to submit the Slurm job with the current values of a and b
        command = f"sbatch --export=seed={seed},n_used={n_used},save={save} b_job_ROCKFISH_gpu.sh"
        
        # Execute the command
        os.system(command)