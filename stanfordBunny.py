# %%
import matplotlib.pyplot as plt
from pytorch3d.io import load_obj
from pytorch3d.ops.subdivide_meshes import SubdivideMeshes
from pytorch3d.structures.meshes import Meshes

# %%
bunny = load_obj("LowPolyStanfordBunny/Bunny-LowPoly.obj", load_textures=False)
sm = SubdivideMeshes()
meshes = Meshes([bunny[0]], [bunny[1].verts_idx])
new_meshes: Meshes = sm(meshes)

new_v = new_meshes.verts_list()[0]
new_f = new_meshes.faces_list()[0]
v = meshes.verts_list()[0]
f = meshes.faces_list()[0]
# %%
view_inits = [(20, 270), (50, 110), (20, 70), (-10, -60), (10, 0)]
for i, el in enumerate(view_inits):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    trisurf = ax.plot_trisurf(
        *v.T,
        triangles=f,
        antialiased=True
    )
    ax.view_init(*el)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'figs_bunny/sb_wo_{i}.svg', transparent=True)
    plt.show()
    plt.close()
# %%
for name, _v, _f in zip(['orig', 'subd'], [v, new_v], [f, new_f]):
    fig = plt.figure(figsize=(20, 20))
    ax = plt.axes(projection='3d')
    trisurf = ax.plot_trisurf(
        *_v.T,
        triangles=_f,
        linewidth=5,
        antialiased=True,
        edgecolor='grey'
    )
    ax.view_init(*view_inits[0])
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(f'figs_bunny/sb_{name}.svg', transparent=True)
    plt.show()
    plt.close()
# %%
