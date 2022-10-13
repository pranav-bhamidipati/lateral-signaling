import json
import pandas as pd
import lateral_signaling as lsig

metadata_fname = lsig.data_dir.joinpath("imaging/FACS_brightfield/metadata.json")
rois_fname = lsig.analysis_dir.joinpath("FACS_brightfield/cell_boundary_vertices.csv")
outdata_fname = lsig.analysis_dir.joinpath("FACS_brightfield/cell_shape_metrics.csv")


def main(
    metadata_fname=metadata_fname,
    rois_fname=rois_fname,
    outdata_fname=outdata_fname,
    area_cutoff=100,
    save=False,
):

    # Get inter-pixel distance in microns
    with metadata_fname.open("r") as f:
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
        print("Writing to:", outdata_fname.resolve().absolute())
        aggdf.to_csv(outdata_fname, index=False)


if __name__ == "__main__":
    main(
        save=True,
    )
