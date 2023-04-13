from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io

from lateral_signaling import __file__ as lsig_file 
from lateral_signaling import data_dir


load_dir = data_dir.joinpath("time_lapse/brightfield_stacks")

def main(frame_skip: int = 5, load_dir: Path = load_dir, save: bool = False, img_fmt: str = "tiff"):

    video_files = sorted(load_dir.glob(f"*.{img_fmt}"))
    n_videos = len(video_files)
    
    im_stack = io.imread(str(video_files[0]))
    n_frames, *imshape = im_stack.shape

    frames_to_process = np.arange(0, n_frames, frame_skip)

    # Get which frames to process
    frame_info = product(enumerate(video_files), frames_to_process)
    frame_info = [(v.name, i, f) for (i, v), f in frame_info]
    queue_df = pd.DataFrame(frame_info, columns=["file", "file_idx", "frame"])

    if save:
        queue_file = Path(lsig_file).parent.joinpath("analyze_PIV_queue.csv")
        print(f"Writing to: {queue_file.resolve().absolute()}")
        queue_df.to_csv(queue_file)

if __name__ == "__main__":
    
    main(
        save=True,
        frame_skip=5,
    )



