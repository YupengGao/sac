import os
import subprocess as sp
import sys

env_list = ['RoboschoolHalfCheetah-v1', 'RoboschoolHopper-v1']

seed_list = [1, 2, 3, 4]

reward_scale_list = [1/0.01, 1/0.1, 1/0.2, 1/0.5]

for env in env_list:

    for seed in seed_list:

        for reward_scale in reward_scale_list:

            log_path = '/dccstor/extrastore/sac_log/' + env + '_' + str(seed) + '_' + str(reward_scale) + '/'
            if not os.path.exists(log_path):
                os.system('mkdir -p ' + log_path)

            cmd = []
            cmd.append('python')
            cmd.append('train.py')
            cmd.append('--seed=' + str(seed))
            cmd.append('--env=' + env)
            cmd.append('--reward_scale=' + str(reward_scale))

            cmd_str = ' '.join(cmd)
            output_path = log_path

            os.system('jbsub -cores 8+1 -mem 64g -queue x86_12h -o ' + output_path + '/' + env + '_' + str(seed) + '_' + str(reward_scale) + '_log.txt ' + cmd_str)

