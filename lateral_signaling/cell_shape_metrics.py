import lateral_signaling as lsig

import os
from glob import glob
import json

import numpy as np
import pandas as pd

# I/O
data_dir       = os.path.abspath("../data/imaging/FACS_brightfield/")
metadata_fname = os.path.join(data_dir, "metadata.json")
rois_fname     = os.path.join(data_dir, "cell_boundary_vertices.csv")
outdata_fname  = os.path.join(data_dir, "cell_shape_metrics.csv")


def main(
    data_dir=data_dir,
    metadata_fname=metadata_fname,
    rois_fname=rois_fname,
    outdata_fname=outdata_fname,
    area_cutoff=100,
    save=False,
):
    
    # Get inter-pixel distance in microns
    with open(metadata_fname, "r") as f:
        m = json.load(f)
        ipd_um = m["interpixel_distance_um"]

    # Read cell boundary data
    df = pd.read_csv(rois_fname, index_col=0)

    # Calculate metrics on data
    aggdfs = []
    for (r, w), d in df.groupby(["roi", "window"]):

        # Extract density condition
        dens = d.density.unique()

        # Get ROI polygon vertices
        verts = d[["x", "y"]].values

        # Calculate metrics, using real units when necessary
        area = lsig.shoelace_area(verts) * (ipd_um ** 2)
        perimeter = lsig.perimeter(verts) * ipd_um
        circularity = lsig.circularity(verts)

        # Filter out ROIs that are too small (erroneous ROI)
        if area < area_cutoff:
            continue
        else:
            aggdfs.append(
                pd.DataFrame(
                    dict(
                        roi=r,
                        window=w,
                        density=dens,
                        area=area,
                        perimeter=perimeter,
                        circularity=circularity,
                    )
                )
            )

    aggdf = pd.concat(aggdfs)
    
    if save:
        print("Writing to:", outdata_fname)
        aggdf.to_csv(outdata_fname, index=False)

main(
    save=True,
)


