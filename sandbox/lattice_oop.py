import numpy as np
import pandas as pd
import scipy.spatial as sp
import scipy.interpolate as snt
import biocircuits
from math import ceil

import os
import glob
import tqdm
import datetime
import time

import colorcet as cc
colors = cc.palette.glasbey_category10
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib import cm

from lattice_signaling import *

############################

def init_uIDs(n, IDsep=":", IDfill="-"):
    """
    Returns a Numpy array of unique ID strings for n cells, at the start of a time-course. 
    The ID consists of 3 components:
    1) Lineage: The first 3 digits are the lineage of the cell. Each of the n cells is 
        treated as a separate lineage at the start of the time-course.
    2) Generation: The next 2 digits are the generation number of the cell (how many 
        divisions it has undergone). This starts at 0 for all cells.
    3) Tree: The last 15 digits are a sequence of binary digits encoding cell lineage. 
        After a cell divides, its daughters enter generation #G. Each daughter is then
        assigned either a 0 or a 1 at digit #G (eg. branches at generation 2 are 
        encoded by the 2nd digit from the right).  
    -  IDsep: the character separating each number
    -  IDfill: If generation G has not arrived yet, this character fills the space.
    """
    
    # Cell IDs, in the same order as their positions in X and S
    return np.array([IDsep.join([str(i).zfill(3), "00", IDfill * 15]) for i in range(n)])


# Given a cell ID and a list of past IDs, find the ID that is the closest ancestor
def cell_ancestor(cell_ID, ID_list, IDsep=":", IDfill="-"):
    """
    """
    if cell_ID in ID_list:
        return cell_ID
    
    # Unpack ID
    lineage, gen, tree = cell_ID.split(IDsep)
    tree = tree.replace(IDfill, '')
    
    # 
    same_lineage = [ID for ID in ID_list if ID.startswith(lineage)]
    for i, _ in enumerate(tree):
        for ID in same_lineage:
            if ID.endswith(tree[i + 1:]):
                return ID
    
    assert False, "cell_ID has no ancestor in ID_list"
    
    
########################
    
class CircularLattice:
    
    def __init__(self, t_points, uIDs_ls, lattice_type, init_state = "blank", IDsep=":", IDfill="-"):
        self.t_points = np.array(t_points).flatten()
        self.uIDs_ls = uIDs_ls
        self.lattice_type = lattice_type  # "static" or "growing"
        self.IDsep = IDsep
        self.IDfill = IDfill
        self.init_state = init_state

    def uIDs(self, t, *args, **kwargs):
        """
        Returns an array of unique IDs of each cell at the given time-point t. 
        Order is preserved between unique IDs, cell coordinates, and other time-varying 
        objects such as the Voronoi object returned by VoronoiLattice.voronoi(t).
        """
        assert (t <= self.t_points[-1]), f"time out of range: lattice not defined at time {t}"
        if "blank".startswith(self.init_state):
            assert (t >= self.t_points[0]), f"time out of range: lattice not defined at time {t}"
        elif "static".startswith(self.init_state):
            t = np.maximum(t, 0)
        else: 
            assert False, "invalid initial state for Lattice object"
        
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        return self.uIDs_ls[int(idx)]

    def n_cells(self, t, *args, **kwargs):
        """
        Returns the number of cells at time t
        """
        return self.uIDs(t).size
                
    def ancestor_uIDs(self, t_past, t_future, kwargs=dict()):
        """        
        Returns an array of cell IDs at the time-point t_future, replacing each cell ID 
        with the ID of its ancestor at time t_past.
        
        Order is preserved between unique IDs, cell coordinates, and other time-varying 
        objects such as the Voronoi object returned by VoronoiLattice.voronoi(t).
        """
        past_IDs = self.uIDs(t_past)
        return np.array(
            [
                cell_ancestor(future_ID, past_IDs, **kwargs)
                for future_ID in self.uIDs(t_future)
            ]
        )

    def map_array(self, t_past, t_future):
        """
        Returns the indices to map an array at time t_past to time t_future.
        Array elements are re-ordered to match the ordering of cells at time 
        t_future and elements are duplicated based on cell division events.
        """
        past_uIDs = self.uIDs(t_past)
        future_uIDs = self.ancestor_uIDs(t_past, t_future)
        mapping = np.concatenate(
            [np.argwhere(past_uIDs == ID).flatten() for ID in future_uIDs]
        )

        return mapping

    def map_matrix(self, t_past, t_future):
        """
        Returns a tuple of indices to map a square matrix at time t_past to 
        time t_future. Matrix rows and columns are re-ordered to match the 
        ordering of cells at time t_future and rows/cols are duplicated 
        based on cell division events.
        """
        mapping = self.map_array(t_past, t_future)
        mapping = np.array([[[x, y] for y in mapping] for x in mapping])

        return mapping[:, :, 0], mapping[:, :, 1]

    def map_array_r(self, t_past, t_future):
        """
        Returns the indices to reverse self.map_array().
        """
        past_uIDs = self.uIDs(t_past)
        future_uIDs = self.ancestor_uIDs(t_past, t_future)
        mapping = np.array(
            [np.argwhere(future_uIDs == ID).flatten()[0] for ID in past_uIDs]
        )

        return mapping

    def map_matrix_r(self, t_past, t_future):
        """
        Returns a tuple of indices to reverse self.map_matrix()
        """
        mapping = self.map_array_r(t_past, t_future)
        mapping = np.array([[[x, y] for y in mapping] for x in mapping])

        return mapping[:, :, 0], mapping[:, :, 1]

    def where_duplicated(self, t_past, t_future):
        """
        Return the indices of duplicated entries in 
            self.ancestor_uIDs(t_past, t_future)
        """
        future_uIDs = self.ancestor_uIDs(t_past, t_future)
        unique = np.zeros(future_uIDs.size, dtype=bool)
        unique[np.unique(future_uIDs, return_index=True)[1]] = True
        return np.nonzero(~unique)[0]

    def get_generations(self, t):
        """Returns the generation number of each cell at time t."""
        return np.array([int(ID.split(sep=self.IDsep)[1]) for ID in self.uIDs(t)])
    
    def dilute_by_division(self, t_past, t_future):
        """
        Returns the fraction by which to dilute each cell's contents based on the number
        of division events between times t_past and t_future. Indexing corresponds to uIDs
        at time t_future.
        """
        if self.n_cells(t_past) == self.n_cells(t_future):
            return np.ones(self.n_cells(t_future), dtype=np.float64)
        else:
            gen_diff = lax.get_generations(t_future) - lax.get_generations(t_past)[lax.map_array(t_past,t_future)]
            return 2. ** (-gen_diff)
        
    def assign_types(self, cell_type_count, method="random", center_type=None, **kwargs):

        points = self.points(self.t_points[0])
        n_points = points.shape[0]
        ct_count = cell_type_count.copy()
        assignments = {}

        if sum(ct_count.values()) < n_points:
            assert (
                sum([x == -1 for x in ct_count.values()]) == 1
            ), "Number of cells in cell_type_numbers does not match the number of points in initial lattice configuration."

            for k in ct_count.keys():
                if ct_count[k] == -1:
                    ct_count[k] = n_points - sum(ct_count.values()) - 1

        elif sum(ct_count.values()) == n_points:
            assert all(
                [val >= 0 for val in ct_count.values()]
            ), "Cannot resolve values in cell_type_numbers with number of points in initial lattice configuration."

        else:
            assert (
                False
            ), "Number of cells in cell_type_dict is greater than number of points in initial lattice configuration."

        if "center".startswith(method):

            assert (center_type is not None), """Missing required argument center_type."""
            
            center_indices = get_center_cells(points, n_center=ct_count[center_type])
            assignments[center_type] = center_indices
            ct_count.pop(center_type)
            
            shuffled_idx = np.delete(np.arange(n_points), center_indices)
            np.random.shuffle(shuffled_idx)
            cumsum = 0
            for k in ct_count.keys():
                assignments[k] = shuffled_idx[slice(cumsum, cumsum + ct_count[k])]
                cumsum += ct_count[k]

        elif "random".startswith(method):
            shuffled_idx = np.arange(n_points)
            np.random.shuffle(shuffled_idx)
            cumsum = 0

            for k in ct_count.keys():
                assignments[k] = shuffled_idx[slice(cumsum, cumsum + ct_count[k])]
                cumsum += ct_count[k]

        self.types = set(assignments.keys())
        self.type_lineages = {typ: np.array([str(i).zfill(3) for i in ids]) for typ, ids in assignments.items()}
    
    def type_indices(self, t, **kwargs):
        """Returns a dictionary of `cell type : indices` pairs at time t."""
        lineages_t = [uid.split(self.IDsep)[0] for uid in self.uIDs(t)]
        type_idx_dict = {typ: np.isin(lineages_t, lins).nonzero() for typ, lins in self.type_lineages.items()}
        return type_idx_dict
    
    def type_array(self, t, **kwargs):
        """Returns a Numpy array of the cell type of each cell at time t."""
        lineages_t = np.array([uid.split(self.IDsep)[0] for uid in self.uIDs(t)])
        type_arr = np.empty_like(lineages_t, dtype=np.dtype('U25'))
        for typ, lins in self.type_lineages.items():
            type_arr[np.isin(lineages_t, lins).nonzero()[0]] = typ
        return type_arr
        
class Regular1DLattice(CircularLattice):
    
    def __init__(self, t_points, n_cells, n_adj, init_state = "static", IDsep=":", IDfill="-"):
        
        self.n_adj = n_adj
        uIDs_ls = [init_uIDs(n_cells, IDsep=IDsep, IDfill=IDfill)]
        
        super().__init__(t_points, uIDs_ls, lattice_type="static", init_state=init_state, IDsep=IDsep, IDfill=IDfill)
        
    def uIDs(self, t, *args, **kwargs):
        return super().uIDs(self.t_points[0])
    
    def n_cells(self, t, *args, **kwargs):
        return super().n_cells(self.t_points[0])
    
    def transition_mtx(self, *args, **kwargs):
        A = np.diag((1,) * (self.n_cells - 1), -1) + np.diag((1,) * (self.n_cells - 1), 1)
        return A / self.n_adj
    
    
class Regular2DLattice(CircularLattice):
    
    def __init__(self, R, r=1, n_adj=6, lattice_type="static", init_state="static", IDsep=":", IDfill="-"):
        
        self.R = R
        self.r = r
        self.X = hex_grid_circle(radius=self.R, r=self.r)
        self.dist = sp.distance.squareform(sp.distance.pdist(self.X))
        
        self.tol = 1e-6
        self.adj = np.array((self.dist <= self.r + self.tol) & (self.dist > 0), dtype=int)
        
        self.n_adj = n_adj
        
        uIDs_ls = [init_uIDs(self.X.shape[0], IDsep=IDsep, IDfill=IDfill)] * 2

        super().__init__(np.array([-np.inf, np.inf]), uIDs_ls, lattice_type=lattice_type, init_state=init_state, IDsep=IDsep, IDfill=IDfill)

    def uIDs(self, *args, **kwargs):
        return super().uIDs(self.t_points[0])

    def n_cells(self, *args, **kwargs):
        return super().n_cells(self.t_points[0])

    def coordinates(self, *args, **kwargs):
        return self.X
    
    def transition_mtx(self, *args, **kwargs):
        return  self.adj / self.n_adj


class VoronoiLattice(CircularLattice):
    
    def __init__(self, t_points, R, sigma, uIDs_ls, voronoi_ls, lattice_type="growing", IDsep=":", IDfill="-", init_state = "static"):
        
        super().__init__(t_points, uIDs_ls, lattice_type, init_state, IDsep, IDfill)
        
        self.R = R
        self.sigma = sigma
        self.voronoi_ls = voronoi_ls
        self.coordinates_ls = [np.array(vor.points) for vor in voronoi_ls]

    def voronoi(self, t):
        """
        Returns the Scipy Voronoi object at the given time-point t.
        Order is preserved between unique IDs, self.coordinates(t), and the Voronoi 
        object returned by self.voronoi(t).
        """
        assert (t <= self.t_points[-1]), (
            f"time out of range: lattice not defined at time {t}"
        )
        if "blank".startswith(self.init_state):
            assert (t >= self.t_points[0]), (
                f"time out of range: lattice not defined at time {t}"
            )
        elif "static".startswith(self.init_state):
            t = np.maximum(t, 0)
        else: 
            assert False, "invalid initial state for Lattice object"
        
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        return self.voronoi_ls[int(idx)]

    def points(self, t):
        """
        Returns the coordinates of cells at the given time-point t.
        Order is preserved between unique IDs, points, and the Voronoi 
        object at time t.
        """
        assert (t <= self.t_points[-1]), (
            f"time out of range: lattice not defined at time {t}"
        )
        if "blank".startswith(self.init_state):
            assert (t >= self.t_points[0]), (
                f"time out of range: lattice not defined at time {t}"
            )
        elif "static".startswith(self.init_state):
            t = np.maximum(t, 0)
        else: 
            assert False, "invalid initial state for Lattice object"
        
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        return self.coordinates_ls[int(idx)]

    def transition_mtx(
        self,
        t,
        t_future=None,
        trans_mtx_func=lattice_adjacency,
        ancestor_kwargs=dict(),
        init_kwargs=dict()
    ):
        """
        Returns the graph transition matrix of the lattice at a given time t. If
        t_future is supplied, the matrix is expanded and to match the lattice shape
        at time t_future for delay calculations.
        """
        if "blank".startswith(self.init_state):
            assert (t >= self.t_points[0]), "time out of range: lattice not defined"
        elif "static".startswith(self.init_state):
            t = np.maximum(t, 0)
        else: 
            assert False, "invalid initial state for Lattice object"

        # If no future time supplied, return the transition matrix
        if t_future is None:
            return lattice_adjacency(self.voronoi(t), R=self.R)

        assert t <= t_future, "t must be less than or equal to t_future"
        assert (t_future >= self.t_points[0]) & (
            t_future <= self.t_points[-1]
        ), "t_future out of range: lattice not defined"

        # Else, calculate the transition matrix and re-map it to its
        #  size at time t_future
        mtx = lattice_adjacency(self.voronoi(t), R=self.R)[self.map_matrix(t, t_future)]

        # Replace duplicated columns with zeros, so duplicated cells do not affect signaling.
        mtx[:, self.where_duplicated(t, t_future)] = 0
        return mtx

        # Note: The choice of which cell in a group of duplicate cells has nonzero entries in its
        # column is arbitrary and should not affect the result. Since the cells undergo numeric
        # integration identically, the result should be the same regardless of which cell is chosen.
    
###################

def lattice_df_to_Voronois(
    df,
    unique_ID_col="cell",
    time_col="time",
    coord_cols=["X_coord", "Y_coord"],
    *args,
    **kwargs
):
    """Return a 3-tuple containing a Numpy array of time-points, a list of lists
    of unique cell IDs at each time-point, and a list of SciPy Voronoi objects 
    at each time-point, according to the Pandas DataFrame argument `df`. 
    ------
    coord_cols should be in order of coordinate axes (i.e. for 2D coordinates, it 
    should be ['X coordinate column name','Y coordinate column name'])
    """
    # Order rows by time, then group by time
    grouped_by_time = df.sort_values(time_col).groupby(time_col)

    # At each time, generate a Voronoi object from the DataFrame
    time_array = np.empty(len(grouped_by_time))
    unique_IDs_list = []
    voronoi_list = []
    for i, tupl in enumerate(grouped_by_time):
        t, df_t = tupl
        time_array[i] = t
        unique_IDs_list.append(df_t.loc[:, unique_ID_col].values)
        voronoi_list.append(sp.Voronoi(df_t.loc[:, coord_cols].values))

    return time_array, unique_IDs_list, voronoi_list


def df_to_VoronoiLattice(df, R, sigma, kwargs=dict()):
    t_points, uIDs_ls, voronoi_ls = lattice_df_to_Voronois(df)
    return VoronoiLattice(
        t_points=t_points,
        R=R,
        sigma=sigma,
        uIDs_ls=uIDs_ls,
        voronoi_ls=voronoi_ls,
        **kwargs
    )

def csv_to_VoronoiLattice(path, R, sigma, csv_kwargs=dict(), lattice_kwargs=dict()):
    df = pd.read_csv(path, **csv_kwargs)
    return df_to_VoronoiLattice(df=df, R=R, sigma=sigma, kwargs=lattice_kwargs)

###################

# class Reaction:

#     def __init__(
#         self, 
#         t_course, 
#         rhs_dict, 
#         E0_dict, 
#         lattice, 
#         cell_type_count=None, 
#         method="random", 
#         center_type=None, 
#         cell_type_kwargs=dict(), 
#         **kwargs
#     ):
        
#         assert (set(rhs_dict.keys()) == lattice.types), (
#             "Must supply rhs func for all cell types in Lattice.types"
#         )
#         assert (set(E0_dict.keys()) == lattice.types), (
#             "Must supply initial value for all cell types in Lattice.types"
#         )

#         if (cell_type_count is not None):
#             lattice.assign_types(cell_type_count, **cell_type_kwargs)

#         self.lattice = lattice
#         self.E0 = E0_dict
#         self.t_course = t_course
#         self.rhs_dict = rhs_dict
#         self.n_species = self.E0[tuple(self.E0.keys())[0]].size

#     def set_params(self, params_dict):
#         """
#         self.params_dict is a dictionary of tuples. For each cell type (key),
#         it stores the parameters for the system of ODEs in a tuple (paired value).
#         """
#         self.params_dict = params_dict

#     def set_inducer(self, func_t, args=(), kwargs=dict()):
#         self.inducer = lambda t: func_t(t, *args, **kwargs)

#     def results_to_df(
#         self, 
#         results=None,
#         time_col="time",
#         species_cols=["expression"],
#         uID_col="unique ID",
#         coord_cols=["X_coord", "Y_coord"],
#     ):
#         """Converts output of Reaction.simulate() to DataFrame."""
        
#         if results is None:
#             results = self.results
#         dfs = []
        
#         for step, time in enumerate(self.t_course):
#             df_dict = {
#                 "step": step,
#                 time_col: time,
#                 uID_col: self.lattice.uIDs(time),
#             }

#             step_data = results[step, self.lattice.map_array_r(time, self.t_course[-1]), :]
            
#             df_dict.update({"type": self.lattice.type_array(time)})
#             df_dict.update({k:v for k, v in zip(coord_cols, self.lattice.points(time).T)})
#             df_dict.update({k:v for k, v in zip(species_cols, step_data.T)})
#             dfs.append(pd.DataFrame(df_dict))

#         return pd.concat(dfs)

#     def integrate_step(self, E, step, end_time, *args, **kwargs):
#         t = self.t_course[step]
#         type_idx_dict = self.lattice.type_indices(t)
#         dE_dt = np.empty_like(E)

#         for typ, rhs in self.rhs_dict.items():
#             dE_dt[type_idx_dict[typ], :] = rhs(
#                 E, 
#                 lattice=self.lattice, 
#                 t=t, 
#                 end_time=end_time, 
#                 params=self.params_dict[typ]
#             )[type_idx_dict[typ], :]

#         return dE_dt

#     def simulate(self, min_val=0, df_kwargs=dict(), progress_bar=False):

#         start = self.t_course[0]
#         end = self.t_course[-1]
            
#         # Get initial expression
#         E = np.empty(
#             (
#                 self.lattice.n_cells(start), 
#                 self.n_species
#             )
#         )

#         for typ, indices in self.lattice.type_indices(start).items():
#             E[indices, :] = self.E0[typ]

#         # Reshape if lattice is growing
#         if "growing".startswith(self.lattice.lattice_type):
#             E = E[self.lattice.map_array(start, end), :]
#         elif "static".startswith(self.lattice.lattice_type):
#             pass
#         else:
#             assert False, f"{type(self.lattice)} object has invalid value for attribute lattice_type."

#         # Perform integration
#         iterator = enumerate(self.t_course[:-1])
#         if progress_bar:
#             iterator = tqdm.tqdm(iterator)
            
#         E_dense = [E]
#         for step, t in iterator:
#             dt = self.t_course[step + 1] - 1
#             dE_dt = self.integrate_step(E, step, end)
#             E = np.maximum(E + dE_dt * dt, min_val)
#             E_dense.append(E)

#         # Output results
#         self.results = np.array(E_dense)
#         self.results_df = self.results_to_df(**df_kwargs)



class Reaction:
    """Signaling reaction on a lattice of cells."""

    def __init__(self):
        pass


# class DelayReaction(Reaction):
#     """Delay differential equation on a lattice."""
    
#     def __init__(
#         self,
#         lattice,
#         dde_rhs = None,
#         E0 = None,
#         delays = (),
#         I_t = None,
#         args = (),
#     ):
#         self.lattice = lattice
#         self.rhs = dde_rhs
#         self.initial = E0
#         self.args = args
#         self.delays = delays
#         self.inducer = I_t
        
#         super().__init__()
    
#     def set_lattice(self, lattice):
#         self.lattice = lattice
    
#     def set_rhs(self, rhs):
#         self.rhs = rhs
    
#     def set_initial_conditions(self, init_func):
#         self.initial = init_func
    
#     def set_args(self, args):
#         self.args = args
    
#     def set_delays(self, delays):
#         self.delays = delays
    
#     def set_inducer(self, ind_func):
#         self.inducer = ind_func
    
#     def simulate(
#         self,
#         t_out,
#         n_time_points_per_step=20,
#         progress_bar=False,
#     ):
#         """Solve a delay differential equation on a growing lattice of cells."""

#         assert all([delay > 0 for delay in self.delays]), "Non-positive delays are not permitted."

#         t0 = t_out[0]
#         t_last = t_out[-1]

#         # Extract shortest and longest non-zero delay parameters
#         min_tau = min(self.delays)

#         # Get graph transition matrix 
#         A = self.lattice.transition_mtx(t0)

#         # Make a shorthand for RHS function
#         def rhs(E, t, E_past):
#             return self.rhs(
#                 E,
#                 t,
#                 E_past,
#                 I_t=self.inducer,
#                 A=A,
#                 delays=self.delays,
#                 params=self.args,
#             )

#         # Define a piecewise function to fetch past values of E
#         time_bins = [t0]
#         E_past_funcs = [lambda t, *args: self.initial(t, I_t=self.inducer, n_cells=self.lattice.n_cells())]

#         def E_past(t):
#             """Define past expression as a piecewise function."""
#             bin_idx = next((i for i, t_bin in enumerate(time_bins) if t < t_bin))
#             return E_past_funcs[bin_idx](t)

#         # Initialize expression.
#         E = self.initial(t0, I_t=self.inducer, n_cells=self.lattice.n_cells())

#         t_dense = []
#         E_dense = []

#         # Integrate in steps of size min_tau. Stops before the last step.
#         t_step = np.linspace(t0, t0 + min_tau, n_time_points_per_step + 1)
#         n_steps = ceil((t_out[-1] - t0) / min_tau)

#         iterator = range(n_steps)
#         if progress_bar:
#             iterator = tqdm.tqdm(iterator)

#         for j in iterator:

#             # Start the next step
#             E_step = [E]

#             # Perform integration
#             for i, t in enumerate(t_step[:-1]):
#                 dE_dt = rhs(E, t, E_past)
#                 dt = t_step[i + 1] - t
#                 E = np.maximum(E + dE_dt * dt, 0)
#                 E_step.append(E)

#             t_dense = t_dense + list(t_step[:-1])
#             E_dense = E_dense + E_step[:-1]

#             # Make B-spline
#             E_step = np.array(E_step)
#             tck = [
#                 [snt.splrep(t_step, E_step[:, cell, i]) for i in range(E.shape[1])]
#                 for cell in range(self.lattice.n_cells())
#             ]

#             # Append spline interpolation to piecewise function
#             time_bins.append(t_step[-1])
#             interp = lambda t, k=j + 1: np.array(
#                 [
#                     [np.maximum(snt.splev(t, tck[cell][i]), 0) for i in range(E.shape[1])]
#                     for cell in range(self.lattice.n_cells())
#                 ]
#             )
#             E_past_funcs.append(interp)

#             # Get time-points for next step
#             t_step += min_tau

#             # Make the last step end at t_last
#             if t_step[-1] > t_last:
#                 t_step = np.concatenate((t_step[t_step < t_last], (t_last,),))

#         # Add data for last time-point
#         t_dense = t_dense + [t_last]
#         E_dense = E_dense + [E]

#         # Interpolate solution and return
#         t_dense = np.array(t_dense)
#         E_dense = np.array(E_dense)

#         E_out = np.empty((len(t_out), *E.shape))
#         for cell in range(E.shape[0]):
#             for i in range(E.shape[1]):
#                 tck = snt.splrep(t_dense, E_dense[:, cell, i])
#                 E_out[:, cell, i] = np.maximum(snt.splev(t_out, tck), 0)

#         self.results = E_out
        
        
class ActiveVoronoi:

    def __init__(self, from_dir, prefix, coord_names = np.array(["X_coord", "Y_coord", "Z_coord"]), IDsep=":", IDfill="-"):
        
        self.dir = from_dir
        self.prefix = prefix
        
        f_npy = glob.glob(os.path.join(from_dir, "*" + prefix + "*.npy"))
        assert len(f_npy) == 1, "prefix matches more than one .npy file in from_dir"
        
        f_npz = glob.glob(os.path.join(from_dir, "*" + prefix + "*.npz"))
        assert len(f_npz) == 1, "prefix matches more than one .npz file in from_dir"
        
        f_npy, f_npz = f_npy[0], f_npz[0]
        
        self.X_arr = np.load(f_npy)
#         self.X_df = pd.read_csv(f_csv)
        
#         self.t_points = np.unique(self.X_df.time.values)
#         self.t0 = self.t_points[0]
        self.n_t = self.X_arr.shape[0]
#         self.dt = self.t_points[1] - self.t_points[0]
        
        self.n_c = self.X_arr.shape[1]
        
#         self.coord_cols = coord_names[np.isin(coord_names, self.X_df.columns)]
#         self.X_arr = self.X_df.loc[:, ["step", *self.coord_cols]].to_numpy()
#         self.X_arr = self.X_arr[np.argsort(self.X_arr[:, 0].flatten())][:, 1:].reshape(self.n_t, self.n_c, self.coord_cols.size)
        
        self.L = ceil(np.max(self.X_arr))
        
        self.IDsep = IDsep
        self.IDfill = IDfill
#         self.uIDs = self.X_df.loc[self.X_df["step"] == 0, "unique_ID"].values
        
        self.types = []
        self.type_n_c = []
        self.type_idx = []
        self.type_lineages = []
        
        with np.load(f_npz, allow_pickle=True) as npz:
            self.l_mtx_list = [npz[str(i)].flat[0] for i in range(self.n_t)]
    
    def transition_mtx(self, t):
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        mtx = self.l_mtx_list[idx]
        return mtx.multiply(1/np.sum(mtx, axis=1))
    
    def X(self, t):
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        return self.X_arr[idx]
    
    
    def set_t_points(self, *args):
        self.t_points = np.linspace(*args)
        self.t0, self.tmax = self.t_points[0], self.t_points[-1]
        self.dt = self.t_points[1] - self.t_points[0]
    
#     def rescale_time(self, t_min, t_max, factor):
#         """
#         Re-scales the time-points in the mechanical (lattice) simulation so
#         that 1 time-unit corresponds to the growth time of the cell line.
#         In other words, ln(2) time-units will be equivalent to the doubling time.
        
#         factor   : float or int 
#             The scaling factor. 
#         """
        
#         self.t0 = t_min
#         self.t_points = np.linspace(t_min * factor, t_max * factor, self.n_t)
#         self.dt = self.t_points[1] - self.t_points[0]
        
        
    def assign_types(self, types, type_n_c, method="center", center_type=None, **kwargs):

        X0 = self.X(self.t0)
        assignments = [None] * len(types)

        if sum(type_n_c) < self.n_c:
            assert (
                sum([x == -1 for x in type_n_c]) == 1
            ), "Number of cells in type_n_c does not match the initial number of cells."

            for k, _ in enumerate(types):
                if type_n_c[k] == -1:
                    type_n_c[k] = self.n_c - sum(type_n_c) - 1

        elif sum(ct_count.values()) == n_points:
            assert all(
                [val >= 0 for val in ct_count.values()]
            ), "Number of cells in type_n_c does not match the initial number of cells."

        else:
            assert (
                False
            ), "Number of cells in type_n_c does not match the initial number of cells."

        if "center".startswith(method):

            assert (center_type is not None), """Missing required argument center_type."""
            
            if isinstance(center_type, str): 
                center_type = next((i for i, k in enumerate(types) if k == center_type))
            
            ci = get_center_cells(X0 - np.mean(X0, axis=0), n_center=type_n_c[center_type])
            
            idx = np.arange(self.n_c)
            idx = np.delete(idx, ci)
            np.random.shuffle(idx)
            
            tnc = np.array(type_n_c)
            tnc[center_type] = 0
            
            assignments = np.split(idx, np.cumsum(tnc))[:-1]
            assignments[center_type] = ci
            
        elif "random".startswith(method):
            
            idx = np.arange(self.n_c)
            np.random.shuffle(idx)
            
            tnc = np.array(type_n_c)
            
            assignments = np.split(idx, np.cumsum(tnc))[:-1]

        self.types = np.array(types)
        self.type_n_c = np.array(type_n_c)
        self.type_idx = assignments
#         self.type_lineages = [self.uIDs[i] for i in self.type_idx]


class DelayReaction(Reaction):
    """Delay differential equation on a lattice."""
    
    def __init__(
        self,
        lattice,
        dde_rhs = None,
        initial = None,
        dde_args = (),
        delay = None,
        inducer = None,
    ):
        self.lattice = lattice
        self.rhs     = dde_rhs
        self.initial = initial
        self.args    = dde_args
        self.delay  = delay
        self.inducer = inducer
        
        self.t_points = lattice.t_points
        self.t0 = lattice.t0
        self.n_t = lattice.n_t
        self.dt = lattice.dt
        
        self.n_c = lattice.n_c
        self.L = lattice.L
        
        self.types = lattice.types
        self.type_idx = lattice.type_idx
        self.type_n_c = lattice.type_n_c
        self.type_lineages = lattice.type_lineages
        
        self.sender_idx = next((self.type_idx[i] for i, k in enumerate(self.types) if k == "sender"))[0]
        self.Sender = np.zeros(self.n_c)
        self.sender_val = 1
        self.Sender[self.sender_idx] = self.sender_val
        
        super().__init__()
    
    
    def simulate(self, progress_bar = False):
        
        max_delay = self.delay
        step_delay = int(max_delay / self.dt)
        
        self.E_save = np.empty((self.n_t, self.n_c))
        E = np.ones(self.n_c) * self.initial
        E[self.sender_idx] = self.sender_val
        
        self.E_save[0] = E
        
        iterator = np.arange(1, self.n_t)
        if progress_bar:
            iterator = tqdm.tqdm(iterator)
        
        for step in iterator:
            
            past_step = int(np.maximum(0, step - step_delay))
            
            dE_dt = self.rhs(
                E, 
                self.E_save[past_step], 
                self.lattice.transition_mtx(self.t_points[past_step]), self.args
            )
            
            dE_dt[self.sender_idx] = 0
            
            E = np.maximum(0, E + dE_dt * self.dt)
            
            self.E_save[step] = E


    def voronoi_finite_polygons_2d(self, vor, radius=None):
        """
        Reconstruct infinite voronoi regions in a 2D diagram to finite
        regions.

        Parameters
        ----------
        vor : Voronoi
            Input diagram
        radius : float, optional
            Distance to 'points at infinity'.

        Returns
        -------
        regions : list of tuples
            Indices of vertices in each revised Voronoi regions.
        vertices : list of tuples
            Coordinates for revised Voronoi vertices. Same as coordinates
            of input vertices, with 'points at infinity' appended to the
            end.

        """

        if vor.points.shape[1] != 2:
            raise ValueError("Requires 2D input")

        new_regions = []
        new_vertices = vor.vertices.tolist()

        center = vor.points.mean(axis=0)
        if radius is None:
            radius = vor.points.ptp().max()

        # Construct a map containing all ridges for a given point
        all_ridges = {}
        for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
            all_ridges.setdefault(p1, []).append((p2, v1, v2))
            all_ridges.setdefault(p2, []).append((p1, v1, v2))

        # Reconstruct infinite regions
        for p1, region in enumerate(vor.point_region):
            vertices = vor.regions[region]

            if all(v >= 0 for v in vertices):
                # finite region
                new_regions.append(vertices)
                continue

            # reconstruct a non-finite region
            ridges = all_ridges[p1]
            new_region = [v for v in vertices if v >= 0]

            for p2, v1, v2 in ridges:
                if v2 < 0:
                    v1, v2 = v2, v1
                if v1 >= 0:
                    # finite ridge: already in the region
                    continue

                # Compute the missing endpoint of an infinite ridge

                t = vor.points[p2] - vor.points[p1]  # tangent
                t /= np.linalg.norm(t)
                n = np.array([-t[1], t[0]])  # normal

                midpoint = vor.points[[p1, p2]].mean(axis=0)
                direction = np.sign(np.dot(midpoint - center, n)) * n
                far_point = vor.vertices[v2] + direction * radius

                new_region.append(len(new_vertices))
                new_vertices.append(far_point.tolist())

            # sort region counterclockwise
            vs = np.asarray([new_vertices[v] for v in new_region])
            c = vs.mean(axis=0)
            angles = np.arctan2(vs[:, 1] - c[1], vs[:, 0] - c[0])
            new_region = np.array(new_region)[np.argsort(angles)]

            # finish
            new_regions.append(new_region.tolist())

        return new_regions, np.asarray(new_vertices)

            
    def plot_vor_colored(self,x,ax,cmap):
        """
        Plot the Voronoi.

        Takes in a set of cell locs (x), tiles these 9-fold, plots the full voronoi, then crops to the field-of-view

        :param x: Cell locations (nc x 2)
        :param ax: matplotlib axis
        """

        L = self.L
        grid_x, grid_y = np.mgrid[-1:2, -1:2]
        grid_x[0, 0], grid_x[1, 1] = grid_x[1, 1], grid_x[0, 0]
        grid_y[0, 0], grid_y[1, 1] = grid_y[1, 1], grid_y[0, 0]
        y = np.vstack([x + np.array([i * L, j * L]) for i, j in np.array([grid_x.ravel(), grid_y.ravel()]).T])

        cmap_print = np.tile(cmap.T,9).T
        bleed = 0.1
        cmap_print = cmap_print[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
        y = y[(y<L*(1+bleed)).all(axis=1)+(y>-L*bleed).all(axis=1)]
        regions, vertices = self.voronoi_finite_polygons_2d(sp.Voronoi(y))


        ax.set(aspect=1,xlim=(0,self.L),ylim=(0,self.L))

        patches = []
        for i, region in enumerate(regions):
            patches.append(Polygon(vertices[region], True,facecolor=cmap_print[i],edgecolor="white",alpha=0.5))

        p = PatchCollection(patches, match_original=True)
        ax.add_collection(p)
        

    def animate(self, n_frames=100, file_name=None, dir_name="plots", cmap = "CET_L8"):
        """
        Animate the simulation, saving to an mp4 file.

        Parameters
        ----------
        n_frames : int
            Number of frames to animate. Spaced evenly throughout **x_save**

        file_name : str
            Name of the file. If **None** given, generates file-name based on the time of simulation

        dir_name: str
            Directory name to save the plot.

        cmap : str
            Name of colormap to use from the colorcet module
        
        """
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            
        print(f"Saving to {dir_name}")
            
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        skip = int((self.lattice.X_arr.shape[0]) / n_frames)
        E_sample = self.E_save[::skip, :]
        E_min, E_max = E_sample.min(), E_sample.max()

        def anim(i):
            ax1.cla()
            colors = cc.cm[cmap](self.normalize(E_sample[i],E_min,E_max))
            self.plot_vor_colored(self.lattice.X_arr[skip * i], ax1, colors)
            ax1.set(aspect=1, xlim=(0, self.L), ylim=(0, self.L))
            ax1.set_title(f"time = {self.lattice.t_points[skip * i]}")

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, bitrate=1800)
        if file_name is None:
            file_name = "animation_%d" % time.time()
        an = animation.FuncAnimation(fig, anim, frames=n_frames, interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)
    
    
    def normalize(self,x, xmin, xmax):
        return (x - xmin)/(xmax - xmin)

