"""
Author: Jacob Heglund
"""

# This script creates a PBS file that runs one hyperameter setting
# on a single node.
import os

# generates a file called 'run.pbs' in the directory this python file is in
pbs_fn = 'run.pbs'
curr_dir = os.getcwd()
pbs_path = os.path.join(curr_dir, pbs_fn)

trainingFilename = 'training_dqn.py'
walltime = '04:00:00'
num_nodes = '1'
cores_per_node = '16'
gpu = 'TeslaM2090'
jobname ='testing_self_play'
netid = 'jheglun2'

directory = curr_dir.replace('\\', '/')

with open(pbs_path, 'w') as f:
    f.write("#!/bin/bash\n")

    f.write("#PBS -N {}\n".format(jobname))
    f.write("#PBS -l walltime={}\n".format(walltime))
    f.write("#PBS -l nodes={}:ppn={}:{}\n".format(num_nodes, cores_per_node, gpu))
    f.write("#PBS -o ~/ctf_public_jh/output.txt\n")
    f.write("#PBS -e ~/ctf_public_jh/error.txt\n")
    f.write("#PBS -q eng-research\n")
    f.write("#PBS -M {}@illinois.edu\n".format(netid))
    f.write("#PBS -m be\n")

    f.write('module load python/3\n')
    f.write('module load anaconda/3\n')
    f.write('module load cuda/9.2\n')
    f.write('source activate /projects/tran-research-group/jheglun2\n')
    f.write("cd ~/ctf_public_jh\n")
    f.write("python {}\n".format(trainingFilename))

