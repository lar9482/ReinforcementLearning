"""
    SOURCE:
    https://stackoverflow.com/questions/66048529/how-to-create-a-heatmap-where-each-cell-is-divided-into-4-triangles
"""

from matplotlib import pyplot as plt
from matplotlib.tri.triangulation import Triangulation
import numpy as np

def create_demo_data(M, N):
    # create some demo data for North, East, South, West
    # note that each of the 4 arrays can be either 2D (N by M) or 1D (N*M)
    # M columns and N rows
    valuesN = np.repeat(np.abs(np.sin(np.arange(N))), M)
    valuesE = np.arange(M * N) / (N * M)
    valuesS = np.random.uniform(0, 1, (N, M))
    valuesW = np.random.uniform(0, 1, (N, M))
    return [valuesN, valuesE, valuesS, valuesW]

def triangulation_for_triheatmap(M, N):
    xv, yv = np.meshgrid(np.arange(-0.5, M), np.arange(-0.5, N))  # vertices of the little squares
    xc, yc = np.meshgrid(np.arange(0, M), np.arange(0, N))  # centers of the little squares
    x = np.concatenate([xv.ravel(), xc.ravel()])
    y = np.concatenate([yv.ravel(), yc.ravel()])
    cstart = (M + 1) * (N + 1)  # indices of the centers

    trianglesN = [(i + j * (M + 1), i + 1 + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesE = [(i + 1 + j * (M + 1), i + 1 + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesS = [(i + 1 + (j + 1) * (M + 1), i + (j + 1) * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    trianglesW = [(i + (j + 1) * (M + 1), i + j * (M + 1), cstart + i + j * M)
                  for j in range(N) for i in range(M)]
    return [Triangulation(x, y, triangles) for triangles in [trianglesN, trianglesE, trianglesS, trianglesW]]

def createHeatMap(M, N, values):

    triangul = triangulation_for_triheatmap(M, N)
    fig, ax = plt.subplots()
    imgs = [
        ax.tripcolor(t, val.ravel(), cmap='RdYlGn', vmin=0, vmax=1, ec='white')
        for t, val in zip(triangul, values)
    ]
    for val, dir in zip(values, [(-1, 0), (0, 1), (1, 0), (0, -1)]):
        for i in range(0, M):
            for j in range(0, N):
                v = 0
                if isinstance(val[0], float):
                    v = val[j * N + i]
                else:
                    v = val[j, i]
                
                ax.text(i + 0.3 * dir[1], j + 0.3 * dir[0], f'{v:.2f}', color='k' if 0.2 < v < 0.8 else 'w', ha='center', va='center')
    cbar = fig.colorbar(imgs[0], ax=ax)

    ax.set_xticks(range(M))
    ax.set_yticks(range(N))
    ax.invert_yaxis()
    ax.margins(x=0, y=0)
    ax.set_aspect('equal', 'box')  # square cells
    plt.tight_layout()
    plt.show()

M, N = 5, 4  # e.g. 5 columns, 4 rows
values = create_demo_data(M, N)
print(values[0])
print('########################')
print(values[1])
print('########################')
print(values[2])
print('########################')
print(values[3])
createHeatMap(M, N, values)