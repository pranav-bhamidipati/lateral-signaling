from functools import partial
from itertools import product
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numba
import numpy as np
from openpiv import preprocess, pyprocess, validation, filters, scaling
import pandas as pd
from psutil import cpu_count
import skimage.io as io


def get_velocity_field(
    image_a,
    image_b,
    window_size=64,
    search_area_size=64,
    overlap=16,
    frame_rate=None,
    time_interval=None,
    scaling_factor=0.37744,
    filter_size=11,
    *args,
    **kwargs,
):
    if time_interval is None:
        assert frame_rate is not None, "Must specify `frame_rate` or `time_interval`."
        time_interval = 1 / frame_rate

    masked_a = preprocess.dynamic_masking(
        image_a,
        filter_size=filter_size,
    )[0]
    masked_b = preprocess.dynamic_masking(
        image_b,
        filter_size=filter_size,
    )[0]

    u, v, sig2noise = pyprocess.extended_search_area_piv(
        masked_a.astype(np.int32),
        masked_b.astype(np.int32),
        window_size=window_size,
        overlap=overlap,
        dt=time_interval,
        search_area_size=search_area_size,
        sig2noise_method="peak2peak",
    )

    x, y = pyprocess.get_coordinates(
        image_size=masked_a.shape, search_area_size=window_size, overlap=overlap
    )

    u, v, mask = validation.global_std(u, v)
    u, v = filters.replace_outliers(u, v, method="localmean", max_iter=3, kernel_size=3)
    x, y, u, v = scaling.uniform(x, y, u, v, scaling_factor=scaling_factor)

    return x, y, u, v


@numba.njit
def compute_speed(u, v):
    return np.sqrt(u.ravel() ** 2 + v.ravel() ** 2)


def get_single_frame_piv_speeds(
    # indices: tuple[int, int],
    # image_stacks: np.ndarray,
    indices: tuple[int, int],
    image_i: np.ndarray,
    image_ip1: np.ndarray,
    save: bool,
    save_dir: Path,
):
    """Calculate the particle speeds for a single frame."""
    video_index, frame = indices
    
    # image_i = image_stacks[image_index, frame]
    # image_ip1 = image_stacks[image_index, frame + 1]
    x, y, u, v = get_velocity_field(image_i, image_ip1, time_interval=1)
    speeds = compute_speed(u, v)
    
    data = pd.DataFrame(dict(video_index=video_index, frame=frame, speed_um_hr=speeds))

    if save:
        fpath = save_dir.joinpath(f"PIV_speeds_video{video_index}_frame{frame}.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        data.to_csv(fpath, index=False)
    else:
        return data


def main(
    load_dir: Path,
    save_dir: Path,
    save: bool = False,
    img_fmt: str = "tiff",
    frame_skip: int = 5,
    nthreads: int = np.inf,
):

    from lateral_signaling import vround

    im_files = np.array(list(load_dir.glob(f"*.{img_fmt}")))
    im_filenames = np.array([f.name for f in im_files])
    image_stacks = np.array([io.imread(str(f)) for f in im_files])

    n_files, n_frames, *imshape = image_stacks.shape
    frames_to_process = np.arange(0, n_frames, frame_skip)
    n_frames_to_process = frames_to_process.size

    # Get which frames to process
    from_frame = product(range(n_files), frames_to_process)
    inputs = (((i, j), image_stacks[i], image_stacks[i, j + 1]) for i, j in from_frame)
    get_speeds = partial(
        get_single_frame_piv_speeds,
        # image_stacks=image_stacks,
        save=save,
        save_dir=save_dir,
    )

    print(
        f"Computing PIV for {n_files * n_frames_to_process} frames "
        f"({n_files} videos, {n_frames_to_process} frames each)."
    )
    
    n_threads = min(nthreads, cpu_count(logical=True))
    print(f"Assembling thread pool of {n_threads} threads.")

    pool = mp.Pool(n_threads)
    speed_data = pool.starmap(get_speeds, inputs)
    pool.close()

    print("Complete.")

    if save:

        speed_df = pd.concat(speed_data, ignore_index=True)
        speed_df["file"] = im_filenames[speed_df.timelapse_idx.values]
        speed_df = (
            speed_df[["file", "frame", "speed_um_hr"]]
            .sort_values(["file", "frame"])
            .reset_index(drop=True)
        )

        fpath = save_dir.joinpath(f"PIV_speed.csv")
        print(f"Writing to: {fpath.resolve().absolute()}")
        speed_df.to_csv(fpath, index=False)


if __name__ == "__main__":

    from lateral_signaling import data_dir, analysis_dir

    main(
        load_dir=data_dir.joinpath("time_lapse"),
        # load_dir=Path(
        #     "/home/pbhamidi/git_data/lateral_signaling/time_lapse_all/brightfield_stacks/"
        # ),
        save_dir=analysis_dir.joinpath("piv"),
        save=True,
        frame_skip=10,
        nthreads=8,
    )
    