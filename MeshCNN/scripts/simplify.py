import os
import trimesh
import open3d
import numpy as np
from tqdm import tqdm
import argparse
from easydict import EasyDict

parser = argparse.ArgumentParser()
parser.add_argument('--folder_in', type=str, help='frequency of showing training results on console')
parser.add_argument('--target_n_faces', type=str, default=5000, help='frequency of saving the latest results')
parser.add_argument('--folder_out', type=str, default='./simplified', help='frequency of saving the latest results')

args = parser.parse_args()

def as_mesh(scene_or_mesh):
    if isinstance(scene_or_mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate([
            trimesh.Trimesh(vertices=m.vertices, faces=m.faces)
            for m in scene_or_mesh.geometry.values()])
    else:
        mesh = scene_or_mesh
    return mesh

def remesh(mesh_orig, target_n_faces):
    if target_n_faces < np.asarray(mesh_orig.triangles).shape[0]:
        mesh = mesh_orig.simplify_quadric_decimation(target_n_faces)
        mesh = mesh.remove_unreferenced_vertices()
    else:
        mesh = mesh_orig
    return mesh

def load_mesh(model_fn):
  # To load and clean up mesh - "remove vertices that share position"
    mesh_ = trimesh.load_mesh(model_fn, process=True)
    mesh_ = as_mesh(mesh_)
    mesh_.remove_duplicate_faces()

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(mesh_.vertices)
    mesh.triangles = open3d.utility.Vector3iVector(mesh_.faces)
    return mesh

def is_mesh(file):
    ext = file[-4:]
    if ext == 'obj':
        return True
    else:
        return False

def main(args):

    if not os.path.exists(args.folder_out):
        os.mkdir(args.folder_out)
    filepaths = os.listdir(args.folder_in)
    for path in tqdm(filepaths):
        path = os.path.join(args.folder_in, path)
        filename, ext = os.path.splitext(os.path.basename(path))
        save_path = os.path.join(args.folder_out, f'{filename}.npz')
        if ext == '.obj':
            mesh_orig = load_mesh(path)

            mesh_sim = remesh(mesh_orig, target_n_faces=args.target_n_faces)
            mesh_sim_dict = EasyDict({'vertices': np.asarray(mesh_sim.vertices), 'faces': np.asarray(mesh_sim.triangles),})
            np.savez(save_path, **mesh_sim_dict)

if __name__ =='__main__':
    main(args)