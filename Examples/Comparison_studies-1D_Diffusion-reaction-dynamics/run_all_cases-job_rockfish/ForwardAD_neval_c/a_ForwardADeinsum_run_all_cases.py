import os

# Define the values
neval_values = [(i, i) for i in [8, 16, 32, 64, 128, 256, 512, 1024]]

# Iterate over each combination
for neval_t,neval_x in neval_values:

    # Construct the directory path
    resultdir = os.path.join(os.getcwd(), 'results', 'ForwardAD_neval_c', 'a_Vanilla-NO_ForwardADeinsum')
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    
    # Construct the command to submit the Slurm job with the current values of a 
    command = f"sbatch --export=neval_t={neval_t},neval_x={neval_x} a_ForwardADeinsum_job_ROCKFISH_gpu.sh"
    
    # Execute the command
    os.system(command)