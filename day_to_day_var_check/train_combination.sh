#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other commands. to ignore just add another # - like ##SBATCH
#SBATCH --partition gtx1080                         ### specify partition name where to run a job. debug: 2 hours; short: 7 days
#SBATCH --time 2-01:00:00                      ### limit the time of job running, partition limit can override this. Format: D-H:MM:SS
#SBATCH --output my_job-id-%J.out                ### output log for running job - %J for job number
##SBATCH --mail-user=gilba@post.bgu.ac.il      ### users email for sending job status
##SBATCH --mail-type=BEGIN,END,FAIL             ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --gres=gpu:1                ### number of GPUs (can't exceed 8 gpus for now) ask for more than 1 only if you can parallelize your code for multi GPU
#SBATCH --nodelist=cs-1080-03  ###,cs-1080-01,cs-1080-02,cs-1080-03,ise-1080-02
### Start you code below ####
module load anaconda              ### load anaconda module
source activate fnet_v1         ### activating environment, environment must be configured before running the job
PYTHONPATH=`pwd` python dowload_and_train.py --experiment_name $1 --exp_type $2 --date_range_indexes $3 && python predict.py --experiment_name $1 ### execute jupyter lab command – replace with your own command e.g. ‘srun --mem=24G python my.py my_arg’ . you may use multiple srun lines, they are the job steps. --mem - the memory to allocate: use 24G x number of allocated GPUs
### --mem=24G
