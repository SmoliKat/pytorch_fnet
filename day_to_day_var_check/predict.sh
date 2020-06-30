#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other commands. to ignore just add another # - like ##SBATCH
#SBATCH --partition gtx1080                         ### specify partition name where to run a job. debug: 2 hours; short: 7 days
#SBATCH --time 2-01:00:00                      ### limit the time of job running, partition limit can override this. Format: D-H:MM:SS
#SBATCH --job-name transfer_learning_LSTM                   ### name of the job
#SBATCH --output my_job-id-%J.out                ### output log for running job - %J for job number
##SBATCH --mail-user=user@post.bgu.ac.il      ### users email for sending job status
##SBATCH --mail-type=BEGIN,END,FAIL             ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1                ### number of GPUs (can't exceed 8 gpus for now) ask for more than 1 only if you can parallelize your code for multi GPU


### Start you code below ####
module load anaconda              ### load anaconda module
source activate fnet_v1         ### activating environment, environment must be configured before running the job
python  predict.py    ### execute jupyter lab command – replace with your own command e.g. ‘srun --mem=24G python my.py my_arg’ . you may use multiple srun lines, they are the job steps. --mem - the memory to allocate: use 24G x number of allocated GPUs
### --mem=24G