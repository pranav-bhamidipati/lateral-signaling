import os
import json

import numpy as np
import pandas as pd

# File containing queue of arguments for array job
q_fname = "lsig_phase_queue.json"

def run_one(config_updates):
    from lsig_phase_run_one import ex
    ex.run(config_updates=config_updates)

with open(q_fname, "r") as f:
    param_list = json.load(f)

# get which job we need to do. The int is necessary because environment variables are stored as strings
run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

# Only use the config update we need to work on
config_updates = param_list[run_id]

# Now run it
run_one(config_updates)

