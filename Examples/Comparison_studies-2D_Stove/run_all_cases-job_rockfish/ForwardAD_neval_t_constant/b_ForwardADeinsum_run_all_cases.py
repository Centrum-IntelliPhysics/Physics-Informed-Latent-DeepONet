import os

# Define the values
neval_x_values = [8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
neval_values = [(20, i) for i in neval_x_values]

# Iterate over each combination
for neval_t,neval_x in neval_values:
        
    # Construct the directory path
    resultdir = os.path.join(os.getcwd(), 'results', 'ForwardAD_neval_t_constant', 'b_Latent-NO_ForwardADeinsum')
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    
    # Construct the command to submit the Slurm job with the current values of a 
    command = f"sbatch --export=neval_t={neval_t},neval_x={neval_x}  b_ForwardADeinsum_job_ROCKFISH_gpu.sh"
    
    # Execute the command
    os.system(command)