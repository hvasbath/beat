from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.cm as cm


def slip_distribution_3d():

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(1, 1, 1, projection="3d")

    cmap = plt.get_cmap("hot")
    pa_col = Poly3DCollection(
        rf_mesh.triangles_xyz,
        cmap=cmap,
        rasterized=True,
    )
    a = inv_slips[0 : rf_mesh.n_triangles]
    #    a = num.random.rand(rf_mesh.n_triangles) - 0.5
    print("a shape", a.shape)
    print("n triangles", rf_mesh.triangles_xyz.shape)
    cbounds = [a.min(), a.max()]
    # norm = matplotlib.colors.Normalize(vmin=a.min(), vmax=a.max(), clip=False)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # colors = [mapper.to_rgba(v) for v in a]
    # print(colors)

    print("cbounds", cbounds)
    pa_col.set_array(a)
    pa_col.set_clim(*cbounds)
    pa_col.set(edgecolor="k", linewidth=0.5, alpha=0.8)
    #    pa_col.autoscale()
    #    print(pa_col.get_array())
    cbs = plt.colorbar(
        pa_col,
        ax=ax,
        orientation="vertical",
        #        cmap=cmap,
    )
    ax.add_collection(pa_col)

    pa_col = Poly3DCollection(
        s_mesh.triangles_xyz,
        cmap=cmap,
        rasterized=True,
    )
    b = inv_slips[-s_mesh.n_triangles : :]
    #    a = num.random.rand(rf_mesh.n_triangles) - 0.5
    print("a shape", b.shape)
    print("n triangles", s_mesh.triangles_xyz.shape)
    cbounds = [b.min(), b.max()]
    # norm = matplotlib.colors.Normalize(vmin=a.min(), vmax=a.max(), clip=False)
    # mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    # colors = [mapper.to_rgba(v) for v in a]
    # print(colors)

    print("cbounds", cbounds)
    pa_col.set_array(b)
    pa_col.set_clim(*cbounds)
    pa_col.set(edgecolor="k", linewidth=0.5, alpha=0.8)
    #    pa_col.autoscale()
    #    print(pa_col.get_array())
    cbs = plt.colorbar(
        pa_col,
        ax=ax,
        orientation="vertical",
        #        cmap=cmap,
    )
    ax.add_collection(pa_col)

    ax.set_xlabel("East- Distance [km]")
    ax.set_ylabel("North- Distance [km]")
    ax.set_zlabel("Depth [km]")
    ax.set_xlim([-3000, 3000])
    ax.set_ylim([-3000, 5000])
    ax.set_zlim([-1000, -5000])
    ax.invert_zaxis()
    fig.tight_layout()

    plt.show()
    fig.savefig("mesh.png", bbox_inches="tight")
