import glob as glob
import numpy as np
import os
import trimesh
from tqdm import tqdm
import open3d

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

def find_neighbor(faces, faces_contain_this_vertex, vf1, vf2, except_face):
    for i in (faces_contain_this_vertex[vf1] & faces_contain_this_vertex[vf2]):
        if i != except_face:
            face = faces[i].tolist()
            face.remove(vf1)
            face.remove(vf2)
            return i

    return except_face


if __name__ == '__main__':

    root = '/content/main/MeshNet/datasets/shrec_21/fold_1'
    new_root = '/content/main/MeshNet/datasets/shrec_21/fold_1_simplified/'
    if not os.path.exists(new_root):
        os.mkdir(new_root)
    for type in os.listdir(root):
        for phrase in ['train', 'test']:
            new_type_path = os.path.join(new_root, type)
            new_phrase_path = os.path.join(new_type_path, phrase)

            old_type_path = os.path.join(root, type)
            old_phrase_path = os.path.join(old_type_path, phrase)

            if not os.path.exists(new_type_path):
                os.mkdir(new_type_path)
            if not os.path.exists(new_phrase_path):
                os.mkdir(new_phrase_path)

            files = glob.glob(os.path.join(old_phrase_path, '*.obj'))
            for file in files:
                # load mesh
                mesh_ = load_mesh(file)
                mesh = remesh(mesh_, target_n_faces=15000)


                # get elements
                vertices = np.asarray(mesh.vertices).copy()
                faces = np.asarray(mesh.triangles).copy()

                # move to center
                center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
                vertices -= center

                # normalize
                max_len = np.max(vertices[:, 0]**2 + vertices[:, 1]**2 + vertices[:, 2]**2)
                vertices /= np.sqrt(max_len)

                # get normal vector
                mesh.compute_triangle_normals()
                face_normal = np.asarray(mesh.triangle_normals).copy()

                # get neighbors
                faces_contain_this_vertex = []
                for i in range(len(vertices)):
                    faces_contain_this_vertex.append(set([]))
                centers = []
                corners = []
                for i in range(len(faces)):
                    [v1, v2, v3] = faces[i]
                    x1, y1, z1 = vertices[v1]
                    x2, y2, z2 = vertices[v2]
                    x3, y3, z3 = vertices[v3]
                    centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
                    corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
                    faces_contain_this_vertex[v1].add(i)
                    faces_contain_this_vertex[v2].add(i)
                    faces_contain_this_vertex[v3].add(i)

                neighbors = []
                for i in range(len(faces)):
                    [v1, v2, v3] = faces[i]
                    n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
                    n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
                    n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
                    neighbors.append([n1, n2, n3])

                centers = np.array(centers)
                corners = np.array(corners)
                faces = np.concatenate([centers, corners, face_normal], axis=1)
                neighbors = np.array(neighbors)

                _, filename = os.path.split(file)
                save_path = os.path.join(new_phrase_path, filename[:-4] + '.npz')
                np.savez(save_path,
                         faces=faces, neighbors=neighbors)

                
