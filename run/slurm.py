"""
Commannd to run code
python run_slurm_parallel.py -c 80 -p "ac_dm_6.py -v -s"
"""
import os
import sys
import argparse
import math
sys.path.append(os.getcwd())
import time
from datetime import timedelta
import numpy as np
from src.utils.run_utils import get_list_pending_experiments
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.file_handling import get_files_recursively
from src.experiment import ExperimentModel

parser = argparse.ArgumentParser()
parser.add_argument("--allocation" ,"-a", type = str, default = 'rrg')

parser.add_argument("--memory", '-mem', type = int, default = 1024) # memory per cpu (in MB)
parser.add_argument('--ntasks', '-n', type=int, default = 16) # number of tasks per CPU
# parser.add_argument("--export", '-e', type = str, default = 'exports3.dat')
parser.add_argument('--pythonfile','-p', type = str)
parser.add_argument('--json', '-j', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('--overwrite','-o', type = bool,   help= "If True, the experiment will overwrite past experiment", default = False)
parser.add_argument('--time' , '-t', type = str, help = "Time wanted to run the jobs. <DD:HH:MM>", default = '00:00:30')

# email for started and completed messages
parser.add_argument("--email", type = str)

# whether we want to estimate the amount of time you need
parser.add_argument("--estimate-time", action='store_true')
parser.add_argument("--estimate-ntasks", action='store_true')

args = parser.parse_args()

experiment_list = args.json
json_files = get_files_recursively(experiment_list)

# Getting the list of python commands needed to be ran
pythoncommands = []
for json_file in json_files:
    print(json_file)
    exp = ExperimentModel.load(json_file)
    if not args.overwrite:
        pending_experiments = get_list_pending_experiments(exp)
    else:
        pending_experiments = list(range(len(exp)))

    num_commands = len(pending_experiments)
    # get the number of nodes that we want
    # nodes = num_commands // cpus
    # if num_commands % cpus == 0:
    #     nodes = nodes
    # else:
    #     nodes = nodes + 1

    for idx in pending_experiments:
        com = 'srun -N1 -n1 --exclusive python ' + args.pythonfile
        # for k in c.keys():
        #     com += ' --{} {}'.format(k, c[k])
        com += f' {json_file} {idx}'
        com+= '\n'
        pythoncommands.append(com)
        
# print(pythoncommands)
print(len(pythoncommands))

num_commands = len(pythoncommands)


if (args.estimate_time):
    print(f"Num experiments: {num_commands}")
    while True:
        try:
            time_per_job = input("Please enter the amount of time needed per job. <D:H:M, H:M, M>: ")
            time_args = time_per_job.split(":")
            time_args = [int(t) for t in time_args]
            num_split = len(time_args)
            
            if num_split < 3:
                time_args = [0] * (3 - num_split) + time_args

            time_per_job = timedelta(days = time_args[0], hours = time_args[1], minutes = time_args[2])

            print(f"Inputted time: {time_per_job.days} days, {time_per_job.seconds//3600} hours, {(time_per_job.seconds//60)%60} minutes")
        except ValueError as e:
            print(f'Sorry, invalid input. Error: {str(e)}')
        else:
            break

    while True:
        try:
            ntasks = int(input("Please enter the number of tasks you wish to have: "))
            if (ntasks <= 0):
                raise ValueError('Bad integer')
        except ValueError as e:
            print(f'Sorry, please enter a valid number of tasks {str(e)}')
        else:
            break

    seconds = time_per_job.seconds * num_commands / ntasks
    time_required = timedelta(seconds = seconds)
    print(f'Estimated time per CPU needed is -t={time_required.days}:{time_required.seconds//3600}:{(time_required.seconds//60)%60}')
    exit()


if (args.estimate_ntasks):
    print(f"Num experiments: {num_commands}")
    while True:
        try:
            time_per_job = input("Please enter the amount of time needed per job. <D:H:M, H:M, M>: ")
            time_args = time_per_job.split(":")
            time_args = [int(t) for t in time_args]
            num_split = len(time_args)
            
            if num_split < 3:
                time_args = [0] * (3 - num_split) + time_args

            time_per_job = timedelta(days = time_args[0], hours = time_args[1], minutes = time_args[2])

            print(f"Inputted time: {time_per_job.days} days, {time_per_job.seconds//3600} hours, {(time_per_job.seconds//60)%60} minutes")
        except ValueError as e:
            print(f'Sorry, invalid input. Error: {str(e)}')
        else:
            break

    while True:
        try:
            time_total = input("Please enter the amount of time you want total. <D:H:M, H:M, M>: ")
            time_args = time_total.split(":")
            time_args = [int(t) for t in time_args]
            num_split = len(time_args)
            
            if num_split < 3:
                time_args = [0] * (3 - num_split) + time_args

            time_total = timedelta(days = time_args[0], hours = time_args[1], minutes = time_args[2])

            print(f"Inputted time: {time_total.days} days, {time_total.seconds//3600} hours, {(time_total.seconds//60)%60} minutes")
        except ValueError as e:
            print(f'Sorry, invalid input. Error: {str(e)}')
        else:
            break

    tasks = math.ceil(time_per_job.seconds * num_commands / time_total.seconds)
    print(f'Estimated number of tasks is -ntasks={tasks}')
    exit()

filename = f'./temp/parallel_scripts/{str(np.random.randint(0,100000))}.txt'

fil = open(filename, 'w')
fil.writelines(pythoncommands)
fil.close()

##########################
# Forming the slurm file #
##########################
 
# Parse time args to form time_str
time_args = args.time.split(":")
time_args = [int(t) for t in time_args]
num_split = len(time_args)

if num_split < 3:
    time_args = [00] * (3 - num_split) + time_args
    
time_str = f'{time_args[0]}-{time_args[1]}:{time_args[2]}:00'

# Forming the rest of the strings needed
allocation_name = args.allocation + '-whitem'
memory = args.memory
email_str = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=ALL\n" if args.email is not None else ""
cwd = os.getcwd()

command = f'parallel --verbose -P -0 -j {args.ntasks} --delay 0.5 :::: ' + filename
print(command)
# sdfsdf

slurm_file = f"#!/bin/sh\n" \
            f"#SBATCH --account={allocation_name}\n" \
            f"#SBATCH --time={time_str}\n" \
            f"#SBATCH --ntasks={args.ntasks}\n" \
            f"#SBATCH --mem-per-cpu={memory}M\n" \
            f"{email_str}" \
            f"source ~/env/bin/activate\n" \
            f"cd {cwd}\n" \
            f"export PYTHONPATH={cwd}:$PYTHONPATH\n" \
            f"{command}"

# Making slurm file
slurm_file_name = f'./temp/slurm_scripts/slurm.sh'
sfile = open(slurm_file_name, 'w')
sfile.write(slurm_file)
sfile.close()
print(slurm_file)

os.system(f'sbatch {slurm_file_name}')
# time.sleep(1)

# make slurm file


    # command_nodes = [ [] for i in range(nodes)]
    # # give each node command
    # for i,c in enumerate(pythoncommands):
    #     command_nodes[i%nodes].append(c)

    # foldername = './temp/parallel_scripts/'
    # if not os.path.exists(foldername):
    #         os.makedirs(foldername, exist_ok=True)
    # filename = [f'./temp/parallel_scripts/node_{i}_{str(args.cpus)}_{str(np.random.randint(0,100000))}.txt' for i in range(nodes) ]

    # # write commands in files
    # for i,f in enumerate(filename):
    #     fil = open(f,'w')
    #     fil.writelines(command_nodes[i])
    #     fil.close()

    # parallel_commands = []
    # for f in filename:
    #     command = 'parallel --verbose -P -0 :::: {}'.format(f)
    #     parallel_commands.append(command)
    # # make slurm file

    # foldername = './temp/slurm_scripts/'
    # if not os.path.exists(foldername):
    #         os.makedirs(foldername, exist_ok=True)
    # slurm_files = [ f'./temp/slurm_scripts/slurm{i}.sh' for i in range(nodes)]
    # cwd = os.getcwd()
    # for n in range(nodes):
    #     slurm_file = f"#!/bin/sh\n" \
    #                 f"#SBATCH --account={allocation_name}\n" \
    #                 f"#SBATCH --time={time_str}\n" \
    #                 f"#SBATCH --ntasks={args.cpus}\n" \
    #                 f"#SBATCH --mem-per-cpu={memory}G\n" \
    #                 f"{email_str}" \
    #                 f"source ~/env/bin/activate\n" \
    #                 f"cd {cwd}\n" \
    #                 f"export PYTHONPATH={cwd}:$PYTHONPATH\n" \
    #                 f"{parallel_commands[n]}"
    #     sfile = open(slurm_files[n], 'w')
    #     sfile.write(slurm_file)
    #     sfile.close()
    #     print(slurm_files[n])
    #     # os.system(f'sbatch {slurm_files[n]}')
    #     time.sleep(1)
