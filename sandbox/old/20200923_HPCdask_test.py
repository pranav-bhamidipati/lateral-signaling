from dask_jobqueue import SLURMCluster
from dask.distributed import Client

# Start in debugging mode
import logging
logging.basicConfig(
    filename='/home/pbhamidi/slurm_out/dasktest.log',
    format='%(levelname)s:%(message)s',
    level=logging.DEBUG,
)


# Set cluster config
cluster = SLURMCluster(
    header_skip=['--mem='],  # ignore --mem option
    project="mthomson",      # #SBATCH -A
    cores=4,
    memory="4G",             # #SBATCH --mem
    job_extra=[              # additional #SBATCH options
        "--mem-per-cpu=1G",
        "-o /home/pbhamidi/slurm_out/dasktest.%N.%j.out",
        "-e /home/pbhamidi/slurm_out/dasktest.%N.%j.err",
    ],
    scheduler_options={
        'host': '172.16.148.22', 
        # 'dashboard_address': ':8000'
    },
)

# Establish client
client = Client(cluster)

# Request two jobs
cluster.scale(2)

