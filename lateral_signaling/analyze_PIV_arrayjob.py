import os
from pathlib import Path
import pandas as pd
from skimage.io import imread
from analyze_PIV import get_single_frame_piv_speeds

queue_file = Path("analyze_PIV_queue.csv")

def main(load_dir: Path, save: bool = False, save_dir: Path = Path(".")):

    # get which job to execute
    run_id = int(os.environ.get("SLURM_ARRAY_TASK_ID"))

    # Only import the images we need to work on
    *_, video_file, video_idx, frame = pd.read_csv(queue_file).iloc[run_id]
    im = imread(str(load_dir.joinpath(video_file)))

    get_single_frame_piv_speeds(
        (video_idx, frame), im[frame], im[frame+1], save=save, save_dir=save_dir
    )

if __name__ == "__main__":
    
    # from lateral_signaling import data_dir, analysis_dir
    data_dir = Path("../data")
    analysis_dir = data_dir.joinpath("analysis")
    
    load_dir = data_dir.joinpath("time_lapse/brightfield_stacks")
    save_dir = analysis_dir.joinpath("piv/raw")

    main(
        load_dir=load_dir,
        save=True,
        save_dir=save_dir,
        # save_dir=Path("/central/home/pbhamidi/git/lateral_signaling/data/analysis/piv"),
    )
