import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import animation
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection
import time
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.patches import Polygon
from scipy.spatial.distance import cdist
from shapely.geometry import Polygon, Point
from descartes import PolygonPatch
from numba import jit
import random

class Cell:
    def __init__(self,id,type):
        self.id = id
        self.type = type

        self.R = []
        self.col = []


class Spherical:
    def __init__(self):
        self.n_cells = []
        self.N_cell_dict = []
        self.cell_ids = []
        self.cells = []
        self.cell_type_list = []
        self.col_dict = []

        self.Ri,self.Rj, self.R = [],[],[]
        self.x, self.x0, self.x_save = [],[],[]


        self.dt = []
        self.tfin = []
        self.t_span = []


        #Params:
        self.eta = 3
        self.tau = 10


    def generate_cells(self,N_cell_dict):
        self.N_cell_dict = N_cell_dict
        k = 0
        id = 0
        for Type,n_i in N_cell_dict.items():
            for i in range(n_i):
                self.cells = self.cells + [Cell(k,Type)]
                self.cell_ids.append(id)
                k += 1
            id +=1
            self.cell_type_list = self.cell_type_list + [Type]
        self.n_cells = k
        self.cell_ids = np.array(self.cell_ids)

    def set_t_span(self,dt=0.05,tfin=50):
        self.dt = dt
        self.tfin = tfin
        self.t_span = np.arange(0, self.tfin, self.dt)
        self.n_T = self.t_span.size

    def set_params(self,W_A, W_P=2.0,W_S = 1):
        if type(W_P) is float:
            self.lambda_P = W_P
        elif type(W_P) is np.ndarray:
            self.lambda_P = np.zeros([self.n_cells])
            for i in range(self.n_cells):
                self.lambda_P[i] = W_P[self.cell_type_list.index(self.cells[i].type)]
        else:
            print("Error, W_P must be float or ndarray")
        if type(W_A) is float:
            self.lambda_A = W_A
        elif type(W_A) is np.ndarray:
            self.lambda_A = np.zeros([self.n_cells, self.n_cells])
            for i in range(self.n_cells):
                for j in range(self.n_cells):
                    if i != j:
                        ci, cj = self.cell_type_list.index(self.cells[i].type), self.cell_type_list.index(
                            self.cells[j].type)
                        self.lambda_A[i, j] = W_A[ci, cj]
                    else:
                        self.lambda_A[i, j] = 0
        else:
            print("Error, W_A must be float or ndarray")

        if type(W_S) is float:
            self.lambda_S = W_S
        elif type(W_S) is np.ndarray:
            self.lambda_S = np.zeros([self.n_cells, self.n_cells])
            for i in range(self.n_cells):
                for j in range(self.n_cells):
                    if i != j:
                        ci, cj = self.cell_type_list.index(self.cells[i].type), self.cell_type_list.index(
                            self.cells[j].type)
                        self.lambda_S[i, j] = W_S[ci, cj]
                    else:
                        self.lambda_S[i, j] = 0
        else:
            print("Error, W_A must be float or ndarray")


    def make_init(self,box_size=12):
        # self.x0 = np.random.uniform(-box_size/2,box_size/2,(self.n_cells,2))
        bound_r = np.sqrt(self.n_cells) * self.R.mean()
        grid = np.unique(np.concatenate([np.arange(0,bound_r,self.R.mean()*np.sqrt(2)),-np.arange(0,bound_r,self.R.mean()*np.sqrt(2))]))
        x,y = np.meshgrid(grid,grid,indexing="ij")

        x,y = x.ravel(),y.ravel()
        mask = (x**2 + y**2 <(bound_r)**2)

        x,y = x[mask],y[mask]
        ind = np.arange(x.size)
        np.random.shuffle(ind)
        ind = ind[:self.n_cells]
        self.x0 = np.array([x[ind],y[ind]]).T


    def define_radii(self,R={"E":1,"T":1}):
        if type(R) is dict:
            self.R = np.zeros([self.n_cells])
            for i in range(self.n_cells):
                self.R[i] = R.index(self.cells[i].type)
            self.Ri, self.Rj = np.meshgrid(R,R,indexing="ij")
        elif type(R) is float:
            self.R = np.ones(self.n_cells)*R
            self.Ri,self.Rj = R,R
        else:
            print("Error, R must be a dict of radii, or a float")

    def V(self,R):
        return (4 / 3) * np.pi * R ** 3

    def A(self,d, Ri, Rj):
        return (np.pi * (Ri ** 2 - ((d ** 2 - Rj ** 2 + Ri ** 2) ** 2) / (4 * d ** 2))) * (d < (Ri + Rj))

    def h_ij(self,d, Ri, Rj):
        return ((Ri - Rj + d) * (Ri + Rj - d)) / (2 * d)

    def v_lens(self,d, Ri, Rj):
        return np.pi / 2 * self.h_ij(d, Ri, Rj) * (3 * Rj - self.h_ij(d, Ri, Rj))

    def Fi(self,d, Ri, Rj, lambd_A, lambd_P,lambda_S):
        return lambda_S*self.A(d, Ri, Rj) * (lambd_A - lambd_P * (
                    (8 * Ri ** 3) / (8 * Ri ** 3 - 3 * self.h_ij(d, Ri, Rj) * (3 * Rj - self.h_ij(d, Ri, Rj)))))


    def pressure(self,d,Ri,Rj):
        return (8 * Ri ** 3) / (8 * Ri ** 3 - 3 * self.h_ij(d, Ri, Rj) * (3 * Rj - self.h_ij(d, Ri, Rj)))

    #
    # def simulate(self):
    #     x_save = np.zeros([self.t_span.size,self.x0.shape[0],self.x0.shape[1]])
    #     x = self.x0.copy()
    #     bound_r = 1.5*np.sqrt(self.n_cells)*self.R.mean()
    #     for i, t in enumerate(self.t_span):
    #         d,disp,ex,ey = get_d_dist(x, self.n_cells)
    #         F_i = Fi(d,self.Ri,self.Rj,self.lambda_A,self.lambda_P,self.lambda_S)
    #         dx = (np.nansum(F_i * -ex,axis=1) + np.random.normal(0,self.eta/np.sqrt(self.dt),self.n_cells))
    #         dy = (np.nansum(F_i * -ey,axis=1) + np.random.normal(0,self.eta/np.sqrt(self.dt),self.n_cells))
    #         x[:,0] = x[:,0] +  dx* self.dt
    #         x[:,1] = x[:,1] +  dy* self.dt
    #         bound_cross = boundary(x,bound_r)
    #         if bound_cross.sum()!=0:
    #             x[bound_cross,0] -= 1*dx[bound_cross]*self.dt
    #             x[bound_cross,1] -= 1*dy[bound_cross]*self.dt
    #         x_save[i] = x.copy()
    #         # print(100*t/self.t_span[-1])
    #     self.x, self.x_save = x,x_save
    #

    def get_persistent_noise(self,dt,eta,n_T,tau,cut_off=1e-3):
        return get_persistent_noise(dt, eta, n_T, tau, cut_off)
        # t_cut_off = int(np.ceil(-np.log(cut_off) * tau/dt))
        # dxy = np.array([np.random.normal(0,eta/np.sqrt(dt),n_T+t_cut_off),
        #                 np.random.normal(0,eta/np.sqrt(dt),n_T+t_cut_off)])
        # Exp = np.exp(-np.arange(t_cut_off) / (tau/dt))
        # RAN = np.array([np.sum(dxy[:,i:t_cut_off+i]*Exp,axis=1)/np.sum(Exp) for i in range(n_T)])
        # return RAN


    def make_noise(self,tau,cut_off = 1e-3):
        self.noise_x,self.noise_y = np.array([self.get_persistent_noise(self.dt,self.eta,self.n_T,tau,cut_off) for i in range(self.n_cells)]).T

    def simulate(self):
        x_save = np.zeros([self.t_span.size,self.x0.shape[0],self.x0.shape[1]])
        x = self.x0.copy()
        bound_r = 10*np.sqrt(self.n_cells)*self.R.mean()*0.7
        self.make_noise(self.tau,1e-3)
        for i, t in enumerate(self.t_span):
            d,disp,ex,ey = get_d_dist(x, self.n_cells)
            F_i = Fi(d,self.Ri,self.Rj,self.lambda_A,self.lambda_P,self.lambda_S) + (d<2)*self.lambda_S*0.25
            Fx, Fy = (np.nansum(F_i * -ex,axis=1)), (np.nansum(F_i * -ey,axis=1))

            # eta_x = (Fx + np.random.normal(0,self.eta/np.sqrt(self.dt),self.n_cells))
            # eta_y = (Fy + np.random.normal(0,self.eta/np.sqrt(self.dt),self.n_cells))
            eta_x = (Fx + self.noise_x[i])
            eta_y = (Fy + self.noise_y[i])
            eta_mod = np.sqrt(eta_x**2 + eta_y**2)
            eta_d = np.random.rayleigh(self.eta/np.sqrt(self.dt))
            dx,dy = Fx + eta_x*eta_d/eta_mod, Fy + eta_y*eta_d/eta_mod,
            x[:,0] = x[:,0] +  dx* self.dt
            x[:,1] = x[:,1] +  dy* self.dt
            bound_cross = boundary(x,bound_r)
            if bound_cross.sum()!=0:
                x[bound_cross] = bounce_edges(x_save[i-1][bound_cross],bound_r,dx*self.dt,dy*self.dt)
            x_save[i] = x.copy()
            # print(100*t/self.t_span[-1])
        self.x, self.x_save = x,x_save

    def set_colors(self,col_dict={"E":"red","T":"blue"}):
        self.col_list = []
        for i in range(self.n_cells):
            cll = self.cells[i]
            cll_type = cll.type
            cll.col = col_dict.get(cll.type)
            self.col_list.append(cll.col)
        self.col_dict = col_dict


    def inspect_out(self,n_plot=5):
        fig, ax = plt.subplots(figsize=(2,n_plot))
        self.plot_all(ax,-1)
        fig.show()

    def plot_all(self,ax1,i):
        ax1.clear()
        ax1.axis('off')
        x = self.x_save[i]
        vor = Voronoi(x)
        regions, vertices = self.voronoi_finite_polygons_2d(vor)
        for id, region in enumerate(regions):
            poly = Polygon(vertices[region])
            circle = Point(x[id]).buffer(self.R[id])
            cell_poly = circle.intersection(poly)
            if cell_poly.area !=0:
                ax1.add_patch(PolygonPatch(cell_poly, ec="white", fc=self.col_list[id]))
            # ax1.add_patch(Polygon(polygon, fill=False, edgecolor="white"))
        ax1.set(xlim=[self.x_save[:,:,0].min() - self.R.max()*2,self.x_save[:,:,0].max()+ self.R.max()*2],ylim=[self.x_save[:,:,1].min() - self.R.max()*2,self.x_save[:,:,1].max()+ self.R.max()*2],aspect=1)

    def normalise(self,x,xmin,xmax):
        return (x-xmin)/(xmax-xmin)

    def plot_var(self, ax1, i,var,vmin,vmax):
        ax1.clear()
        ax1.axis('off')
        x = self.x_save[i]
        vor = Voronoi(x)
        regions, vertices = self.voronoi_finite_polygons_2d(vor)
        cmap = plt.cm.plasma(self.normalise(var,vmin,vmax))
        for id, region in enumerate(regions):
            poly = Polygon(vertices[region])
            circle = Point(x[id]).buffer(self.R[id])
            cell_poly = circle.intersection(poly)
            if cell_poly.area != 0:
                ax1.add_patch(PolygonPatch(cell_poly, ec="white", fc=cmap[id]))
            # ax1.add_patch(Polygon(polygon, fill=False, edgecolor="white"))
        ax1.set(xlim=[self.x_save[:, :, 0].min() - self.R.max() * 2, self.x_save[:, :, 0].max() + self.R.max() * 2],
                ylim=[self.x_save[:, :, 1].min() - self.R.max() * 2, self.x_save[:, :, 1].max() + self.R.max() * 2],
                aspect=1)

    def get_pressures(self):
        p = np.zeros([self.x_save.shape[0],self.x_save.shape[1]])
        for i, x in enumerate(self.x_save):
            d,disp,ex,ey = get_d_dist(x, self.n_cells)
            p[i] = np.nansum(self.pressure(d,self.Ri,self.Rj)*(d<(self.Ri+self.Rj)),axis=1)
        self.p = p


    def animate(self,n_frames = 100,file_name=None, dir_name="plots", xlim=None, ylim=None, quiver=False, voronoi=False,
                **kwargs):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        skip = int((self.x_save.shape[0])/n_frames)
        def animate(i):
            self.plot_all(ax1,skip*i)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)
        if file_name is None:
            file_name = "animation %d" % time.time()
        an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)


    def animate_pressures(self,n_frames = 100,file_name=None, dir_name="plots", xlim=None, ylim=None, quiver=False, voronoi=False,
                **kwargs):
        self.get_pressures()
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)

        skip = int((self.x_save.shape[0])/n_frames)
        pmin,pmax = self.p[::skip].min(),self.p[::skip].max()
        def animate(i):
            self.plot_var(ax1,skip*i,self.p[i*skip],pmin,pmax)

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, bitrate=1800)
        if file_name is None:
            file_name = "animation %d" % time.time()
        an = animation.FuncAnimation(fig, animate, frames=n_frames, interval=200)
        an.save("%s/%s.mp4" % (dir_name, file_name), writer=writer, dpi=264)



    def voronoi_finite_polygons_2d(self,vor, radius=None):
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

    def circles(self,x, y, s, c='b', ax=None, vmin=None, vmax=None, **kwargs):
        """
        Make a scatter of circles plot of x vs y, where x and y are sequence
        like objects of the same lengths. The size of circles are in data scale.

        Parameters
        ----------
        x,y : scalar or array_like, shape (n, )
            Input data
        s : scalar or array_like, shape (n, )
            Radius of circle in data unit.
        c : color or sequence of color, optional, default : 'b'
            `c` can be a single color format string, or a sequence of color
            specifications of length `N`, or a sequence of `N` numbers to be
            mapped to colors using the `cmap` and `norm` specified via kwargs.
            Note that `c` should not be a single numeric RGB or RGBA sequence
            because that is indistinguishable from an array of values
            to be colormapped. (If you insist, use `color` instead.)
            `c` can be a 2-D array in which the rows are RGB or RGBA, however.
        vmin, vmax : scalar, optional, default: None
            `vmin` and `vmax` are used in conjunction with `norm` to normalize
            luminance data.  If either are `None`, the min and max of the
            color array is used.
        kwargs : `~matplotlib.collections.Collection` properties
            Eg. alpha, edgecolor(ec), facecolor(fc), linewidth(lw), linestyle(ls),
            norm, cmap, transform, etc.

        Returns
        -------
        paths : `~matplotlib.collections.PathCollection`

        Examples
        --------
        a = np.arange(11)
        circles(a, a, a*0.2, c=a, alpha=0.5, edgecolor='none')
        plt.colorbar()

        License
        --------
        This code is under [The BSD 3-Clause License]
        (http://opensource.org/licenses/BSD-3-Clause)
        """

        if np.isscalar(c):
            kwargs.setdefault('color', c)
            c = None
        if 'fc' in kwargs: kwargs.setdefault('facecolor', kwargs.pop('fc'))
        if 'ec' in kwargs: kwargs.setdefault('edgecolor', kwargs.pop('ec'))
        if 'ls' in kwargs: kwargs.setdefault('linestyle', kwargs.pop('ls'))
        if 'lw' in kwargs: kwargs.setdefault('linewidth', kwargs.pop('lw'))

        patches = [Circle((x_, y_), s_) for x_, y_, s_ in np.broadcast(x, y, s)]
        collection = PatchCollection(patches, **kwargs)
        if c is not None:
            collection.set_array(np.asarray(c))
            collection.set_clim(vmin, vmax)
        if ax is None:
            ax = plt.gca()
        ax.add_collection(collection)
        ax.autoscale_view()
        if c is not None:
            plt.sci(collection)
        return collection


@jit(cache=True, nopython=True)
def h_ij(d, Ri, Rj):
    return ((Ri - Rj + d) * (Ri + Rj - d)) / (2 * d)

@jit(cache=True, nopython=True)
def get_d_dist(x,n_cells):
    disp = np.zeros((n_cells,2,n_cells))
    X_,Y_ = np.outer(x[:,0],np.ones(n_cells)), np.outer(x[:,1],np.ones(n_cells))
    disp[:,0,:] = X_ - X_.T
    disp[:,1,:] = Y_ - Y_.T
    d = np.sqrt(disp[:,0,:]**2 + disp[:,1,:]**2)
    ex, ey = disp[:, 0, :] / d, disp[:, 1, :] / d
    return d, disp, ex, ey

@jit(cache=True, nopython=True)
def boundary(x,r):
    return x[:,0]**2 + x[:,1]**2 > r**2
