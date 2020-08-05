import numpy as np
import pandas as pd
import scipy.spatial as sp
import scipy.interpolate as snt
import biocircuits

import os
import tqdm
import datetime
from math import ceil

import colorcet as cc
colors = cc.palette.glasbey_category10

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


class DelayReaction(Reaction):
    """Delay differential equation on a lattice."""
    
    def __init__(
        self,
        lattice,
        dde_rhs = None,
        E0 = None,
        delays = (),
        I_t = None,
        args = (),
    ):
        self.lattice = lattice
        self.rhs = dde_rhs
        self.initial = E0
        self.args = args
        self.delays = delays
        self.inducer = I_t
        
        super().__init__()
    
    def set_lattice(self, lattice):
        self.lattice = lattice
    
    def set_rhs(self, rhs):
        self.rhs = rhs
    
    def set_initial_conditions(self, init_func):
        self.initial = init_func
    
    def set_args(self, args):
        self.args = args
    
    def set_delays(self, delays):
        self.delays = delays
    
    def set_inducer(self, ind_func):
        self.inducer = ind_func
    
    def simulate(
        self,
        t_out,
        n_time_points_per_step=20,
        progress_bar=False,
    ):
        """Solve a delay differential equation on a growing lattice of cells."""

        assert all([delay > 0 for delay in self.delays]), "Non-positive delays are not permitted."

        t0 = t_out[0]
        t_last = t_out[-1]

        # Extract shortest and longest non-zero delay parameters
        min_tau = min(self.delays)

        # Get graph transition matrix 
        A = self.lattice.transition_mtx()

        # Make a shorthand for RHS function
        def rhs(E, t, E_past):
            return self.rhs(
                E,
                t,
                E_past,
                I_t=self.inducer,
                A=A,
                delays=self.delays,
                params=self.args,
            )

        # Define a piecewise function to fetch past values of E
        time_bins = [t0]
        E_past_funcs = [lambda t, *args: self.initial(t, I_t=self.inducer, n_cells=self.lattice.n_cells())]

        def E_past(t):
            """Define past expression as a piecewise function."""
            bin_idx = next((i for i, t_bin in enumerate(time_bins) if t < t_bin))
            return E_past_funcs[bin_idx](t)

        # Initialize expression.
        E = self.initial(t0, I_t=self.inducer, n_cells=self.lattice.n_cells())

        t_dense = []
        E_dense = []

        # Integrate in steps of size min_tau. Stops before the last step.
        t_step = np.linspace(t0, t0 + min_tau, n_time_points_per_step + 1)
        n_steps = ceil((t_out[-1] - t0) / min_tau)

        iterator = range(n_steps)
        if progress_bar:
            iterator = tqdm.tqdm(iterator)

        for j in iterator:

            # Start the next step
            E_step = [E]

            # Perform integration
            for i, t in enumerate(t_step[:-1]):
                dE_dt = rhs(E, t, E_past)
                dt = t_step[i + 1] - t
                E = np.maximum(E + dE_dt * dt, 0)
                E_step.append(E)

            t_dense = t_dense + list(t_step[:-1])
            E_dense = E_dense + E_step[:-1]

            # Make B-spline
            E_step = np.array(E_step)
            tck = [
                [snt.splrep(t_step, E_step[:, cell, i]) for i in range(E.shape[1])]
                for cell in range(self.lattice.n_cells())
            ]

            # Append spline interpolation to piecewise function
            time_bins.append(t_step[-1])
            interp = lambda t, k=j + 1: np.array(
                [
                    [np.maximum(snt.splev(t, tck[cell][i]), 0) for i in range(E.shape[1])]
                    for cell in range(self.lattice.n_cells())
                ]
            )
            E_past_funcs.append(interp)

            # Get time-points for next step
            t_step += min_tau

            # Make the last step end at t_last
            if t_step[-1] > t_last:
                t_step = np.concatenate((t_step[t_step < t_last], (t_last,),))

        # Add data for last time-point
        t_dense = t_dense + [t_last]
        E_dense = E_dense + [E]

        # Interpolate solution and return
        t_dense = np.array(t_dense)
        E_dense = np.array(E_dense)

        E_out = np.empty((len(t_out), *E.shape))
        for cell in range(E.shape[0]):
            for i in range(E.shape[1]):
                tck = snt.splrep(t_dense, E_dense[:, cell, i])
                E_out[:, cell, i] = np.maximum(snt.splev(t_out, tck), 0)

        self.results = E_out
        
        
class ActiveVoronoi:
    
    def __init__(self, active_vor, IDsep=":", IDfill="-"):
        
        assert active_vor.t_span.size == active_vor.x_save.shape[0], (
            """Number of time points must match shape of active_vor.x_save"""
        )
        
        self.t_points = active_vor.t_span
        self.n_t = active_vor.t_span.size
        self.t0 = active_vor.t_span[0]
        self.X_arr = active_vor.x_save
        self.n_c = active_vor.x_save.shape[1]
        self.ndim = active_vor.x_save.shape[2]
        self.l_int = active_vor.l_save #interface length
        self.IDsep = IDsep
        self.IDfill = IDfill
        self.uIDs = init_uIDs(active_vor.n_c, IDsep=IDsep, IDfill=IDfill)

    def transition_mtx(self, t):
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        return self.l_arr[idx] / np.sum(self.l_arr[idx], axis=1)[:, np.newaxis]
    
    def X(self, t):
        idx = np.searchsorted(self.t_points, t, side="right") - 1
        return self.X_arr[idx]
    
    def save_coords(self, fname, index=False, df_kwargs=dict(), csv_kwargs=dict()):
        # Store coordinate data in dict
        data = dict(
            step      = np.repeat(np.arange(self.n_t), self.n_c),
            time      = np.repeat(self.t_points, self.n_c),
            unique_ID = np.tile(self.uIDs, self.n_t),
        )

        coords = ('X_coord', 'Y_coord', 'Z_coord')
        for i in range(self.ndim):
            data[coords[i]] = self.X_arr[:, :, i].flatten()
        
        pd.DataFrame(data, **df_kwargs).to_csv(fname, index=index, **csv_kwargs)
        
    def save_l_int(self, fname, index=True, series_kwargs=dict(), csv_kwargs=dict()):
        """Save a sparse, long version of self.l__arr in a MultiIndexed Series"""
        midx = pd.MultiIndex.from_arrays(np.array(self.l_arr.nonzero()))
        l_series = pd.Series(
            arr[arr.nonzero()], 
            index=midx,
            **series_kwargs,
        )
        
        l_series.to_csv(fname, index=index, **csv_kwargs)
    
    def save_l_int2(self, to_dir):
        """
        Save l_int (pairwise length of interfaces) to file to_dir.npz.
        """
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)

        np.savez(to_dir, **{str(step): csr for step, csr in zip(range(self.n_t), self.l_int)})
    
    def save_all(self, to_dir, prefix):
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)

        self.save_coords(os.path.join(to_dir, prefix + "_cell_coords.csv"))
        self.save_l_int2(os.path.join(to_dir, prefix + "_l_sparse"))