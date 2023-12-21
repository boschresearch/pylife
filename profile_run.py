import pylife.mesh.gradient
import numpy as np
import pandas as pd

def create_mesh(n):
    x_list = np.linspace(0, 1, n)
    y_list = np.linspace(0, 1, n)
    z_list = np.linspace(0, 1, n)
    X,Y,Z = np.meshgrid(x_list, y_list, z_list)
    h = 1/n

    df_node_mesh = pd.DataFrame({"x": X.flatten(), "y": Y.flatten(), "z": Z.flatten()})

    df_node_mesh.x += (2*np.random.random(size=(n**3))-1)*h*0.4
    df_node_mesh.y += (2*np.random.random(size=(n**3))-1)*h*0.4
    df_node_mesh.z += (2*np.random.random(size=(n**3))-1)*h*0.4

    df_node_mesh["mises"] = np.random.random(size=(n**3))
    df_node_mesh.reset_index(names=["node_id"], inplace=True)

    # add elements
    df_list = []
    for k in range(n-1):
        for j in range(n-1):
            for i in range(n-1):
                element_id = k*(n-1)*(n-1) + j*(n-1) + i
                nodes = [
                    df_node_mesh.iloc[k*n*n + j*n + i,:],
                    df_node_mesh.iloc[k*n*n + j*n + i+1,:],
                    df_node_mesh.iloc[k*n*n + (j+1)*n + i+1,:],
                    df_node_mesh.iloc[k*n*n + (j+1)*n + i,:],
                    df_node_mesh.iloc[(k+1)*n*n + j*n + i,:],
                    df_node_mesh.iloc[(k+1)*n*n + j*n + i+1,:],
                    df_node_mesh.iloc[(k+1)*n*n + (j+1)*n + i+1,:],
                    df_node_mesh.iloc[(k+1)*n*n + (j+1)*n + i,:]
                ]
                df = pd.concat(nodes, axis=1).T
                df["element_id"] = element_id
                df_list.append(df)
    df_mesh = pd.concat(df_list)
    df_mesh.set_index(["element_id", "node_id"], inplace=True)
    df_mesh.index = df_mesh.index.set_levels(df_mesh.index.levels[0].astype(int), level=0)
    df_mesh.index = df_mesh.index.set_levels(df_mesh.index.levels[1].astype(int), level=1)
    
    return df_mesh

import timeit

def evaluate(n):
    df_mesh = create_mesh(n)

    tstart = timeit.default_timer()
    # given a mesh in `pylife_mesh` with column `mises`, compute the gradient
    gradient = df_mesh.gradient_3D.gradient_of('mises')
    
    tend = timeit.default_timer()

    return n**3, tend - tstart

evaluate(10)