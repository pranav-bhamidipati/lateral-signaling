# Dominik's script for parallelization on HPC

import os
import json
import sacred
from glob import glob
import dask
from dask.distributed import Client, progress
import time
import datetime

def run_one(queue_dir):
    from experiment import ex
    if (not os.path.isfile(f'{queue_dir}/config.json')) or (not os.path.isfile(f'{queue_dir}/run.json')):
        print('Did not find the required configuration files... Aborting...')
        return

    with open(f'{queue_dir}/run.json','r') as f:
        if json.load(f)['status'] == "COMPLETED":
            print('Run already completed... Aborting...')
            return

    config = sacred.config.load_config_file(f'{queue_dir}/config.json')
    ex.run(config_updates=config)
    print(f'finished {queue_dir}')


@dask.delayed
def wait_5(i):
    print(f'{i}: {datetime.datetime.now()}')
    time.sleep(5)
    print(f'{i}: {datetime.datetime.now()}')


if __name__ == '__main__':
    lazy_result = []
    #n_slurm_tasks = int(os.environ['SLURM_NTASKS'])
    n_slurm_tasks =2 
    client = Client(threads_per_worker=1, n_workers=n_slurm_tasks)
    for folder in glob('queue/*'):
        lazy_result.append(dask.delayed(run_one)(folder))
    dask.compute(*lazy_result)