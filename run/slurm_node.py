"""
Commannd to run code
python run_slurm_parallel.py -c 80 -p "ac_dm_6.py -v -s"
"""
import os
import sys
import argparse
sys.path.append(os.getcwd())
import time
import numpy as np
from src.utils.run_utils import get_list_pending_experiments
from src.utils.json_handling import get_sorted_dict, get_param_iterable
from src.utils.file_handling import get_files_recursively

parser = argparse.ArgumentParser()
parser.add_argument("--allocation" ,"-a", type = str, default = 'rrg')

parser.add_argument("--cpus", '-c', type = int, default = 16)
parser.add_argument("--memory", '-mem', type = int, default = 32)
parser.add_argument('--ntasks', '-n', type=int, default = 4) # number of tasks per CPU
# parser.add_argument("--export", '-e', type = str, default = 'exports3.dat')
parser.add_argument('--pythonfile','-p', type = str)
parser.add_argument('--json', '-j', type = str ,nargs='+', help='Json Files', required=True) # the json configuration
parser.add_argument('--overwrite','-o', type = bool,   help= "If True, the experiment will overwrite past experiment", default = False)
parser.add_argument('--days' , '-d', type = int, help = "Number of days to run the code", default = 0)
# works only for upto 24 hours
parser.add_argument("--hours", '-hr', type = int, default = 0)
parser.add_argument("--minutes", '-m', type = int, default = 30)

# email for started and completed messages
parser.add_argument("--email", type = str)

args = parser.parse_args()

experiment_list = args.json
json_files = get_files_recursively(experiment_list)
for json_file in json_files:
    print(json_file)
    d = get_sorted_dict(json_file)
    experiments = get_param_iterable(d)
    if not args.overwrite:
        pending_experiments = get_list_pending_experiments(experiments)
    else:
        pending_experiments = list(range(len(experiments)))

    print(f"Experiments : {json_file} : {pending_experiments}")
    # by default use only one node
    allocation_name = args.allocation + '-whitem'
    time_str = "{}-{}:{}:00".format(str(args.days) , str(args.hours), str(args.minutes))
    cpus = args.cpus * args.ntasks
    memory = args.memory


    num_commands = len(pending_experiments)
    # get the number of nodes that we want
    nodes = num_commands // cpus
    if num_commands % cpus == 0:
        nodes = nodes
    else:
        nodes = nodes + 1


    pythoncommands = []
    for idx in pending_experiments:
        com = 'python ' + args.pythonfile
        # for k in c.keys():
        #     com += ' --{} {}'.format(k, c[k])
        com += f' {json_file} {idx}'
        com+= '\n'
        pythoncommands.append(com)

    command_nodes = [ [] for i in range(nodes)]
    # give each node command
    for i,c in enumerate(pythoncommands):
        command_nodes[i%nodes].append(c)

    foldername = './temp/parallel_scripts/'
    if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)
    filename = [f'./temp/parallel_scripts/node_{i}_{str(args.cpus)}_{str(np.random.randint(0,100000))}.txt' for i in range(nodes) ]

    # write commands in files
    for i,f in enumerate(filename):
        fil = open(f,'w')
        fil.writelines(command_nodes[i])
        fil.close()

    parallel_commands = []
    for f in filename:
        command = 'parallel --verbose -P -0 :::: {}'.format(f)
        parallel_commands.append(command)
    # make slurm file

    foldername = './temp/slurm_scripts/'
    if not os.path.exists(foldername):
            os.makedirs(foldername, exist_ok=True)
    slurm_files = [ f'./temp/slurm_scripts/slurm{i}.sh' for i in range(nodes)]
    cwd = os.getcwd()
    email_str = f"#SBATCH --mail-user={args.email}\n#SBATCH --mail-type=ALL\n" if args.email is not None else ""
    for n in range(nodes):
        slurm_file = f"#!/bin/sh\n" \
                    f"#SBATCH --account={allocation_name}\n" \
                    f"#SBATCH --time={time_str}\n" \
                    f"#SBATCH --ntasks={args.cpus}\n" \
                    f"#SBATCH --mem={memory}G\n" \
                    f"#SBATCH --nodes=1\n" \
                    f"{email_str}" \
                    f"source ~/env/bin/activate\n" \
                    f"cd {cwd}\n" \
                    f"export PYTHONPATH={cwd}:$PYTHONPATH\n" \
                    f"{parallel_commands[n]}"
        sfile = open(slurm_files[n], 'w')
        sfile.write(slurm_file)
        sfile.close()
        os.system(f'sbatch {slurm_files[n]}')
        time.sleep(1)
