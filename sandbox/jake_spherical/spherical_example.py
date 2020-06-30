from spherical_model import Spherical
import numpy as np
import time

sp = Spherical()
sp.generate_cells({"E":40, "T":40})
sp.set_params(W_A = np.array([[1.5,1.3],[1.3,1.5]]), W_P = 1.0, W_S = 0.4)
sp.define_radii(R = 1.0)
sp.make_init(box_size=1.5*np.sqrt(sp.n_cells)/np.pi)
sp.eta = 0.225
sp.set_t_span(dt = 0.1,tfin=500)
sp.tau = 25
sp.simulate()
sp.set_colors(col_dict={"E":"red","T":"blue"})
sp.inspect_out()
# sp.animate_pressures(voronoi=True,n_frames=100)
sp.animate(voronoi=True,n_frames=100)

#
#
#
#
# sp = Spherical()
# sp.generate_cells({"E":15,"T":15,"X":15})
# sp.set_params(W_A = np.array([[1.5,1.3,1.3],
#                               [1.3,1.5,1.3],
#                               [1.3,1.3,1.15]]),W_P = 1.0,W_S = 0.4)
# sp.define_radii(R=1.0)
# sp.make_init(box_size=1.5*np.sqrt(sp.n_cells)/np.pi)
# sp.eta =0.225
# sp.set_t_span(dt = 0.1,tfin=2000)
# sp.tau = 25
# sp.simulate()
# sp.set_colors(col_dict={"E":"red","T":"blue","X":"green"})
# sp.inspect_out()
# sp.animate(voronoi=True,n_frames=100)


#
# from spherical_model import get_d_dist
# def get_av_self_self(x):
#     I, J = np.meshgrid(sp.cell_ids,sp.cell_ids,indexing="ij")
#     d,__,__,__ = get_d_dist(x,sp.n_cells)
#     neighbour_mask = (d<2)*(d>0)
#     return np.sum(neighbour_mask*(I==J))/np.sum(neighbour_mask)
#
# #
# # fig, ax = plt.subplots(figsize=(5,4))
# # # sp.tau = 25
# # # sp.simulate()
# # avss = np.array([get_av_self_self(x) for x in sp.x_save])
# # ax.plot(avss)
# # # sp.tau = sp.dt
# # # sp.simulate()
# # # avss = np.array([get_av_self_self(x) for x in sp.x_save])
# # # ax.plot(avss)
# # ax.set(xlabel="Time",ylabel="% of neighbours that are self-self")
# # fig.show()
# #
# #
# # from scipy.spatial import Voronoi
# # x = sp.x
# # # x = x[x[:,0]>0]
# # # x = x[x[:,1]>0]
# # # x = x[x[:,0]<4]
# # # x = x[x[:,1]<4]
# #
# # x = np.zeros([50,2])
# # x[:10,0] = np.arange(0,15,1.5)
# # x[10:20,0] = np.arange(0.75,15.75,1.5)
# # x[20:30,0] = np.arange(0,15,1.5)
# # x[30:40,0] = np.arange(0.75,15.75,1.5)
# # x[40:50,0] = np.arange(0,15,1.5)
# # x[10:20,1] = np.sqrt(3)*0.75
# # x[20:30,1] = np.sqrt(3)*0.75*2
# # x[30:40,1] = np.sqrt(3)*0.75*3
# # x[40:50,1] = np.sqrt(3)*0.75*4
# #
# #
# #
# # t0 = time.time()
# # for i in range(int(1e4)):
# #     vor = Delaunay(x)
# #     # get_d_dist(sp.x, sp.n_cells)
# # t1 = time.time()
# # print(t1-t0)
# #
# #
# # regions, vertices = sp.voronoi_finite_polygons_2d(vor)
# #
# #
# #
# # from matplotlib.patches import Polygon
# # from scipy.spatial.distance import cdist
# # from shapely.geometry import Polygon, Point
# # from descartes import PolygonPatch
# # fig, ax = plt.subplots()
# # for i in range(50):
# #
# #     region = regions[i]
# #     poly = Polygon(vertices[region])
# #     circle = Point(x[i]).buffer(0.9)
# #     cell_poly = circle.intersection(poly)
# #     if cell_poly.area != 0:
# #         ax.add_patch(PolygonPatch(poly, ec="white", fc="blue"))
# #         ax.add_patch(PolygonPatch(circle, ec="white", fc="white",alpha=0.5,zorder=100))
# #     # ax1.add_patch(Polygon(polygon, fill=False, edgecolor="white"))
# # ax.set(xlim=(-2,18),ylim=(-2,18),aspect=1)
# # fig.show()
# #
#
# x = sp.x
# x = x[x[:,0]<0]
# x = x[x[:,1]<0]
# x = x[x[:,0]>-4]
# x = x[x[:,1]>-4]
#
# d, disp, ex, ey = get_d_dist(x, x.shape[0])
#
# from numba import jit
#
# @jit(nopython=True,cache=True)
# def circle_intersection_(x1,y1,x2,y2,d,R1,R2):
#     """for cell i xi2[i],yi2[i] is the start going clockwise of the lens prescribed. yi1,yi2 is the end of the lens"""
#     c1 = (R1**2 - R2**2)/(2*d**2)
#     c2 = 0.5*np.sqrt(2*(R1**2 + R2**2)/d**2 - ((R1**2 - R2**2)**2)/d**4 - 1)
#     xi1 = 0.5*(x1+x2) + c1*(x2-x1) + c2*(y2-y1)
#     xi2 = 0.5*(x1+x2) + c1*(x2-x1) - c2*(y2-y1)
#     yi1 = 0.5*(y1+y2) + c1*(y2-y1) + c2*(x1-x2)
#     yi2 = 0.5*(y1+y2) + c1*(y2-y1) - c2*(x1-x2)
#     return xi1,yi1,xi2,yi2
#
#
# def circle_intersection(x1,y1,x2,y2,d,R1,R2):
#     mask = (d>0)&(d<2)
#     xi1,xi2,yi1,yi2 = np.ones_like(x1)*np.nan,np.ones_like(x1)*np.nan,np.ones_like(x1)*np.nan,np.ones_like(x1)*np.nan
#     xi1[mask],xi2[mask],yi1[mask],yi2[mask] = circle_intersection_(x1[mask], y1[mask], x2[mask], y2[mask], d[mask], R1[mask], R2[mask])
#     return xi1,xi2,yi1,yi2
#
# from scipy import sparse
#
# x1,x2 = np.meshgrid(x[:,0],x[:,0],indexing="ij")
# y1,y2 = np.meshgrid(x[:,1],x[:,1],indexing="ij")
# R1,R2 = np.meshgrid(np.ones(x.shape[0]),np.ones(x.shape[0]))
# xi1, yi1,xi2,yi2 = circle_intersection(x1, y1, x2, y2, d, R1, R2)
#
# t0 = time.time()
# for i in range(int(1e4)):
#     xi,yi,xi2,yi2 = circle_intersection(x1,y1,x2,y2,d,R1,R2)
# t1 = time.time()
# print(t1-t0)
# # yi1 = yi1[~np.isnan(xi1)]
# # xi1 = xi1[~np.isnan(xi1)]
# # yi2 = yi2[~np.isnan(xi2)]
# # xi2 = xi2[~np.isnan(xi2)]
#
# xi1, yi1,xi2,yi2 = circle_intersection(x1, y1, x2, y2, d, R1, R2)
#
# N =x.shape[0]
# cmap = plt.cm.rainbow(np.arange(N)/N)
# fig, ax = plt.subplots()
# for i in range(N):
#     sp.circles(x[i,0],x[i,1],1,ax=ax,alpha=0.4,color=cmap[i])
#     if i == 1:
#         ax.scatter(xi[i],yi[i],color=cmap[i])
#         ax.scatter(xi2[i],yi2[i],color="black",alpha=0.7)
#         # ax.quiver(xi[i],yi[i],-sv_x[i],-sv_y[i])
#
# # ax.scatter(xi2,yi2,alpha=0.4)
# ax.set(aspect=1)
# fig.show()
#
# @jit(cache=True,nopython=True)
# def get_overlap_(xi1,xi2,yi1,yi2,xi3,xi4,yi3,yi4):
#     m1 = (yi2-yi1)/(xi2-xi1)
#     m2 = (yi4-yi3)/(xi4-xi3)
#     x = (yi1 - yi3 +xi3*m2 - xi1*m1)/(m2-m1)
#     y = yi1 + m1*(x-xi1)
#     return x,y
#
#
# @jit(nopython=True,cache=True)
# def get_3(d,i):
#     D = d<2
#     out = np.zeros((d.shape[0],d.shape[1]),dtype=np.bool_)
#     out[D[i]] = D[D[i]]
#     out2 = out*out.T
#     J= np.empty((int(np.sum(out2/2))),dtype=np.int64)
#     K = np.zeros((int(np.sum(out2/2))),dtype=np.int64)
#     l = 0
#     for j in range(d.shape[0]):
#         for k in range(d.shape[1]):
#             if j>k:
#                 if out2[j,k]:
#                     J[l],K[l] = j,k
#                     l = l+1
#     return J,K
#
#
# #
# # def get_overlap(xi1,xi2,yi1,yi2,d,x,R):
# #     d[d==0] = np.nan
# #     Ox,Oy = np.ones_like(xi1)*np.nan,np.ones_like(xi1)*np.nan
# #     for i in range(xi1.shape[0]):
# #         J,K = get_3(d,i)
# #         for l in range(J.size):
# #             j,k = J[l],K[l]
# #             x1, y1 = xi1[i, j], yi1[i, j]
# #             x2,y2 = xi2[i,j],yi2[i,j]
# #             x3,y3,x4,y4 = xi1[i,k],yi1[i,k],xi2[i,k],yi2[i,k] #possibly...
# #             Ox[i,j],Oy[i,j] = get_overlap_(x1,x2,y1,y2,x3,x4,y3,y4)
# #             xo,yo = get_overlap_(x1,x2,y1,y2,x3,x4,y3,y4)
# #     return Ox,Oy
#
# @jit(nopython=True,cache=True)
# def get_extra_area_length(xc,yc, xo,yo,x1,x2,x3,x4,y1,y2,y3,y4,R):
#     v1,v2,v3,v4 = np.array([x1-xo,y1-yo]),np.array([x2-xo,y2-yo]),np.array([x3-xo,y3-yo]),np.array([x4-xo,y4-yo])
#     l1,l2,l3,l4 = np.linalg.norm(v1),np.linalg.norm(v2),np.linalg.norm(v3),np.linalg.norm(v4)
#     if l1<l2:
#         sinth = np.cross(np.dot(v1,v4)/(l1*l4))
#         if sinth > 0:
#             # th = np.arccos(np.dot(v4,v1)/(l1*l4))
#             phi = np.arccos(np.dot(np.array([x1-xc,y1-yc]),np.array([x4-xc,y4-yc])/R))
#             # sinth = np.sin(th)
#             extra_area = 1/2*(R**2 * (phi - np.sin(phi)) + sinth*(l1*l4))
#             la,lb = l1,l4
#         else:
#             extra_area = np.nan
#             la,lb = np.nan,np.nan
#     else:
#         sinth = np.cross(v3,v2)/(l2*l3)
#         if sinth > 0:
#             # th = np.arccos(np.dot(v2,v3)/(l2*l3))
#             phi = np.arccos(np.dot(np.array([x2-xc,y2-yc]),np.array([x3-xc,y3-yc]))/R**2)
#             # sinth = np.sin(th)
#             extra_area = 1/2*(R**2 * (phi - np.sin(phi)) + sinth*(l2*l3))
#             la,lb = l2,l3
#         else:
#             extra_area = np.nan
#             la, lb = np.nan, np.nan
#     return extra_area,la,lb
#
# @jit(nopython=True,cache=True)
# def chord_length(d,Ri,Rj):
#     return (1/d)*(np.sqrt((-d + Rj - Ri)*(-d - Rj + Ri)*(-d + Rj + Ri)*(d+Rj + Ri)))
#
# @jit(nopython=True,cache=True)
# def lens_vol(d,Ri,Rj):
#     d2 = (d**2 + Rj**2 - Ri**2)/(2*d)
#     dV = Ri**2 * np.arccos(d2/Ri) - d2*np.sqrt(Ri**2 - d2**2)
#     n = d.shape[0]
#     dv_out = np.empty(n)
#     for i in range(n):
#         dv_out[i] = np.nansum(dV[i])
#     return dv_out
#
# @jit(nopython=True,cache=True)
# def get_x12y12(x):
#     x1,y1 = np.outer(x[:,0],np.ones(x.shape[0])), np.outer(x[:,1],np.ones(x.shape[0]))
#     x2,y2 = x1.T,y1.T
#     return x1,x2,y1,y2
#
#
# def get_area_length(x,Ri,Rj,d):
#     """I think a problem is that the geometires don't click. Need to explicitly measure nearest circle point.???????"""
#     x1,x2,y1,y2 = get_x12y12(x)
#     xi1, yi1, xi2, yi2 = circle_intersection(x1, y1, x2, y2, d, Ri, Rj)
#     d[d==0] = np.nan
#     L,dV = get_area_length_(x, Ri, Rj, d, xi1, yi1, xi2, yi2)
#     return L,dV
#
# @jit(nopython=True,cache=True)
# def get_area_length_(x,Ri,Rj,d,xi1, yi1, xi2, yi2):
#     """I think a problem is that the geometires don't click. Need to explicitly measure nearest circle point.???????"""
#     L = chord_length(d,Ri,Rj)
#     dV = lens_vol(d,Ri,Rj)
#     for i in range(xi1.shape[0]):
#         J,K = get_3(d,i)
#         for l in range(J.size):
#             j,k = J[l],K[l]
#             x1, y1 = xi1[i, j], yi1[i, j]
#             x2,y2 = xi2[i,j],yi2[i,j]
#             x3,y3,x4,y4 = xi1[i,k],yi1[i,k],xi2[i,k],yi2[i,k]
#             xo,yo = get_overlap_(x1,x2,y1,y2,x3,x4,y3,y4)
#             extra_area, l1, l4 = get_extra_area_length(x[i,0],x[i,1],xo, yo, x1,x2,x3,x4,y1, y2, y3, y4, Ri[i,j])
#             dV[i] = dV[i] - extra_area
#             L[i,j] -= l1
#             L[i,k] -= l4
#     return L, dV
#
# t0 = time.time()
# for i in range(int(1e5)):
#     L, dV = get_area_length(x,R1,R2,d)
# t1 = time.time()
# print(t1-t0)
#
# xi1, yi1, xi2, yi2 = circle_intersection(x1, y1, x2, y2, d, R1, R2)
#
# Ox,Oy = get_overlap(xi1,xi2,yi1,yi2,d,x,np.ones(x.shape[0]))
#
# N =x.shape[0]
# cmap = plt.cm.rainbow(np.arange(N)/N)
# fig, ax = plt.subplots()
# for i in range(N):
#     sp.circles(x[i,0],x[i,1],1,ax=ax,alpha=0.4,color=cmap[i])
#     ax.text(x[i,0],x[i,1],i,color="k",fontsize=15)
#     if i == 0:
#         for j in range(xi.shape[1]):
#             ax.text(xi[i,j], yi[i,j],j, color=cmap[i])
#             ax.text(xi2[i,j], yi2[i,j],j, color="black", alpha=0.7)
#             ax.text(Ox[i,j], Oy[i,j], j,color="red")
# ax.set(aspect=1)
# fig.show()

