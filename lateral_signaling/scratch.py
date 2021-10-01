
def plot_var(
    ax,
    i,
    X,
    var,
    cell_radii=None,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    ec=None,
    ifcc="",
    ppatch_kwargs=dict(edgecolor='gray'),
    lcoll_kwargs=dict(),
    title=None,
    axis_off=True,
    xlim=(),
    ylim=(),
    aspect=None,
    plot_ifc=True,
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    ax.clear()
    if axis_off:
        ax.axis("off")
        
    vor = Voronoi(X)
    regions, vertices = voronoi_finite_polygons_2d(vor)
    
    if cell_radii is None:
        cell_radii = np.ones(vor.npoints) * 5
    
    if vmin is None:
        vmin = var.min()
    if vmax is None:
        vmax = var.max()
    
    if type(cmap) is str:
        cmap_ = cc.cm[cmap]
    else:
        cmap_ = cmap
    cols = cmap_(normalize(var, vmin, vmax))
    
    sender_col = cc.cm[sender_clr[0]](sender_clr[1] / 256)
    for j, region in enumerate(regions):
        poly = Polygon(vertices[region])
        circle = Point(X[j]).buffer(cell_radii[j])
        cell_poly = circle.intersection(poly)
        if cell_poly.area != 0:
            if j in sender_idx:
                ax.add_patch(PolygonPatch(cell_poly, fc=sender_col, **ppatch_kwargs))
            else:
                ax.add_patch(PolygonPatch(cell_poly, fc=cols[j], **ppatch_kwargs))
    
    if plot_ifc:
        pts = [Point(*x).buffer(rad) for x, rad in zip(X, cell_radii)]
        Adj = make_Adj(int(np.sqrt(X.shape[0])))
        infcs = []
        for i, j in zip(*Adj.nonzero()):
            if (j - i > 0) & (pts[i].intersects(pts[j])):
                infc = pts[i].boundary.intersection(pts[j].boundary)
                infcs.append(infc)
        ax.add_collection(LineCollection(infcs, color=ifcc, **lcoll_kwargs))
    
    if not xlim:
        xlim=[X[:, 0].min(), X[:, 0].max()]
    if not ylim:
        ylim=[X[:, 1].min(), X[:, 1].max()]
    if aspect is None:
        aspect=1
    ax.set(
            xlim=xlim,
            ylim=ylim,
            aspect=aspect,
        )
    
    if title is not None:
        ax.set_title(title)

    if extend is None:
        
        # Extend colorbar if necessary
        n = var.shape[0]        
        ns_mask = ~ np.isin(np.arange(n), sender_idx)
        is_under_min = var.min(initial=0.0, where=ns_mask) < vmin
        is_over_max  = var.max(initial=0.0, where=ns_mask) > vmax
        extend = ("neither", "min", "max", "both")[is_under_min + 2 * is_over_max]
                
    if colorbar:
        
        # Construct colorbar
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=mpl.colors.Normalize(vmin, vmax), 
                cmap=cmap_), 
            ax=ax,
            aspect=cbar_aspect,
            extend=extend,
            **cbar_kwargs
        )


def inspect_hex(
    X,
    var_t,
    ax=None,
    idx=-1,
    cell_radii=None,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    #     ec="red",
    ifcc="red",
    ppatch_kwargs=dict(edgecolor="gray"),
    lcoll_kwargs=dict(),
    plot_ifc=False,
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    
    nt = var_t.shape[0]
    k = np.arange(nt)[idx]
    
    if X.ndim == 2:
        Xk = X.copy()
    else:
        Xk = X[k]
    
    if cell_radii is None:
#         Xk_centered = Xk - Xk.mean(axis=0)
#         max_rad = np.linalg.norm(Xk_centered, axis=1).max() * 2
#         crk = max_rad * np.ones(Xk.shape[0], dtype=np.float32) 
        crk = np.ones(X.shape[0]) * 5
        
    elif cell_radii.ndim == 2:
        crk = cell_radii[k]
        
    else:
        crk = cell_radii.copy()
    
    plot_var(
        ax,
        k,
        Xk,
        var_t[k],
        cell_radii=crk,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        ifcc=ifcc,
        ppatch_kwargs=ppatch_kwargs,
        lcoll_kwargs=lcoll_kwargs,
        plot_ifc=plot_ifc,
        sender_idx=sender_idx,
        sender_clr=sender_clr,
        colorbar=colorbar,
        cbar_aspect=cbar_aspect,
        cbar_kwargs=cbar_kwargs,
        extend=extend,
       **kwargs
    )
    
    
    
    ################################
    
    ax,
    X,
    var,
    n_interp=120,
    vmin=None,
    vmax=None,
    cmap="CET_L8",
    title=None,
    axis_off=True,
    xlim=(),
    ylim=(),
    aspect=None,
    pcolormesh_kwargs=dict(),
    sender_idx=np.array([], dtype=int),
    sender_clr=("bmw", 150),
    sender_kwargs=dict(),
    colorbar=False,
    cbar_aspect=20,
    cbar_kwargs=dict(),
    extend=None,
    **kwargs
