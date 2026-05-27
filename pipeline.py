#!/usr/bin/env python3
"""
Forest Reconstruction Pipeline
===============================
PointTree (segmentation) -> AdTree (reconstruction)

Usage:
    python pipeline.py --build                      # compile AdTree once
    python pipeline.py --input forest.laz
    python pipeline.py --input forest.laz --scan-type ULS
"""

import os, sys, json, shutil, subprocess, argparse, time, glob, io
import numpy as np

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ADTREE_SRC    = os.path.join(BASE_DIR, 'AdTree-main')
ADTREE_BUILD  = os.path.join(BASE_DIR, 'AdTree-build')
ADTREE_BIN    = os.path.join(BASE_DIR, 'adtree_path.txt')
POINTTREE_OUT = os.path.join(BASE_DIR, 'output', 'pointtree')
ADTREE_OUT    = os.path.join(BASE_DIR, 'output', 'adtree')

SCAN_TYPE           = 'TLS'
ENABLE_NOISE_FILTER = False
NB_NEIGHBORS        = 20
STD_RATIO           = 2.0
VOXEL_SIZE          = 0.05
FACES_PER_LEAF      = 12
KEEP_RATIOS         = [0.6, 0.3]


def apply_patches(adtree_src):
    skel = os.path.join(adtree_src, 'AdTree', 'skeleton.cpp')
    with open(skel) as f:
        content = f.read()
    content = content.replace('int density = ceil(random_float() * 10);', 'int density = ceil(random_float() * 1);')
    content = content.replace('generate_leaves(currentLeafVertex, 0.05);', 'generate_leaves(currentLeafVertex, 0.02);')
    content = content.replace('double radius = 0.2 / log((float)num_edges(simplified_skeleton_));', 'double radius = 0.04 / log((float)num_edges(simplified_skeleton_));')
    content = content.replace('\tdouble epsiony = 0.02;\n', '\tdouble epsiony = 0.10;\n')
    print('  Patches A + epsiony applied')
    old_generate = ('void Skeleton::generate_leaves(SGraphVertexDescriptor i_LeafVertex, double leafsize_Factor)\n{\n\t//generate a random density number\n    int density = ceil(random_float() * 1);\n    double radius = 0.04 / log((float)num_edges(simplified_skeleton_));\n\t//get the position of the current leaf vertex and its parent\n    vec3 pCurrent = simplified_skeleton_[i_LeafVertex].cVert;\n    SGraphVertexDescriptor i_LeafParent = simplified_skeleton_[i_LeafVertex].nParent;\n    vec3 pParent = simplified_skeleton_[i_LeafParent].cVert;\n\t//get the end position where the leaf should grow\n    vec3 pEnd = pCurrent - (random_float() / 2.0) * ((pCurrent - pParent).normalize());\n\n\t//generate i-th random leaf\n\tfor (int i = 0; i < density; ++i)\n\t{\n\t\t//generate a random leaf position\n        vec3 dirLeaf((random_float() - 0.5) / 0.5, (random_float() - 0.5) / 0.5, (random_float() - 0.5) / 0.5);\n\t\tdirLeaf = dirLeaf.normalize();\n        double l = random_float() * radius;\n\t\tvec3 pLeaf = pEnd + dirLeaf * l;\n\t\t//generate normal and color vector\n\t\tvec3 dirParent2Leaf = (pLeaf - pParent).normalize();\n\t\tvec3 normal = (cross(dirParent2Leaf, dirLeaf)).normalize();\n\t\t//generate a new leaf\n\t\tLeaf newleaf;\n\t\tnewleaf.cPos = pLeaf;\n\t\tnewleaf.cDir = dirLeaf;\n\t\t//generate a random normal vector direction\n        vec3 delta((random_float() - 0.5) / 0.5, (random_float() - 0.5) / 0.5, (random_float() - 0.5) / 0.5);\n        newleaf.cNormal = (normal + random_float()*delta*0.5).normalize();\n\t\tnewleaf.pSource = i_LeafVertex;\n\t\tnewleaf.nLength = BoundingDistance_ * leafsize_Factor;\n\t\tnewleaf.nRad = newleaf.nLength / 5;\n\t\tVecLeaves_.push_back(newleaf);\n\t}\n\n\treturn;\n}')
    new_generate = ('void Skeleton::generate_leaves(SGraphVertexDescriptor i_LeafVertex, double leafsize_Factor)\n{\n\t//generate a random density number\n    int density = ceil(random_float() * 1);\n    double radius = 0.04 / log((float)num_edges(simplified_skeleton_));\n\t//get the position of the current leaf vertex and its parent\n    vec3 pCurrent = simplified_skeleton_[i_LeafVertex].cVert;\n    SGraphVertexDescriptor i_LeafParent = simplified_skeleton_[i_LeafVertex].nParent;\n    vec3 pParent = simplified_skeleton_[i_LeafParent].cVert;\n\t// branch growth direction\n    vec3 branchDir = (pCurrent - pParent).normalize();\n\n\t//generate i-th random leaf\n\tfor (int i = 0; i < density; ++i)\n\t{\n\t\t// leaf base directly at branch tip\n        double offset = random_float() * radius * 0.5;\n        vec3 pLeaf = pCurrent - branchDir * offset;\n\n\t\t// leaf direction: perpendicular away from branch\n        vec3 randPerp((random_float()-0.5)/0.5, (random_float()-0.5)/0.5, (random_float()-0.5)/0.5);\n        randPerp = randPerp.normalize();\n        randPerp = (randPerp - branchDir * dot(randPerp, branchDir)).normalize();\n        vec3 dirLeaf = (randPerp * 0.6f + branchDir * 0.4f).normalize();\n\n\t\t//generate normal and color vector\n\t\tvec3 dirParent2Leaf = (pLeaf - pParent).normalize();\n\t\tvec3 normal = (cross(dirParent2Leaf, dirLeaf)).normalize();\n\t\t//generate a new leaf\n\t\tLeaf newleaf;\n\t\tnewleaf.cPos = pLeaf;\n\t\tnewleaf.cDir = dirLeaf;\n\t\t//generate a random normal vector direction\n        vec3 delta((random_float()-0.5)/0.5, (random_float()-0.5)/0.5, (random_float()-0.5)/0.5);\n        newleaf.cNormal = (normal + random_float()*delta*0.5).normalize();\n\t\tnewleaf.pSource = i_LeafVertex;\n\t\tnewleaf.nLength = BoundingDistance_ * leafsize_Factor;\n\t\tnewleaf.nRad = newleaf.nLength / 5;\n\t\tVecLeaves_.push_back(newleaf);\n\t}\n\n\treturn;\n}')
    assert old_generate in content, 'ERROR: generate_leaves not found!'
    content = content.replace(old_generate, new_generate, 1)
    print('  Patch C applied')
    old_func = ('bool Skeleton::reconstruct_leaves(SurfaceMesh *mesh) {\n    if (!add_leaves())\n        return false;\n\n    if (VecLeaves_.empty())\n        return false;\n\n    for (std::size_t i = 0; i < VecLeaves_.size(); i++) {\n        const Leaf& iLeaf = VecLeaves_[i];\n        //compute the center and major axis, minor axis of the leaf quad\n        vec3 pCenter((iLeaf.cPos + (0.5 * iLeaf.cDir * iLeaf.nRad)));\n        vec3 dirMajor(0.5 * iLeaf.cDir * iLeaf.nLength);\n        vec3 dirMinor(0.5 * cross(iLeaf.cDir, iLeaf.cNormal)*iLeaf.nRad);\n        //compute the corner coordinates\n        const vec3 a = pCenter - dirMajor - dirMinor;\n        const vec3 b = pCenter + dirMajor - dirMinor;\n        const vec3 c = pCenter + dirMajor + dirMinor;\n        const vec3 d = pCenter - dirMajor + dirMinor;\n        SurfaceMesh::Vertex va = mesh->add_vertex(a);\n        SurfaceMesh::Vertex vb = mesh->add_vertex(b);\n        SurfaceMesh::Vertex vc = mesh->add_vertex(c);\n        SurfaceMesh::Vertex vd = mesh->add_vertex(d);\n        mesh->add_triangle(va, vb, vc);\n        mesh->add_triangle(va, vc, vd);\n    }\n\n    return true;\n}')
    new_func = ('bool Skeleton::reconstruct_leaves(SurfaceMesh *mesh) {\n    if (!add_leaves())\n        return false;\n\n    if (VecLeaves_.empty())\n        return false;\n\n    const int nSegs = 6;\n    for (std::size_t i = 0; i < VecLeaves_.size(); i++) {\n        const Leaf& iLeaf = VecLeaves_[i];\n        vec3 dir = iLeaf.cDir; vec3 norm = iLeaf.cNormal;\n        vec3 axisL = dir.normalize() * iLeaf.nLength;\n        vec3 axisW = cross(dir, norm).normalize() * iLeaf.nRad;\n        vec3 pBase = iLeaf.cPos;\n        std::vector<SurfaceMesh::Vertex> leftVerts, rightVerts;\n        for (int s = 0; s <= nSegs; ++s) {\n            double t = (double)s / nSegs;\n            double width = sin(M_PI * t) * (2.0 - 0.3 * t);\n            vec3 pAlong = pBase + axisL * t;\n            leftVerts.push_back(mesh->add_vertex(pAlong - axisW * width * 0.5));\n            rightVerts.push_back(mesh->add_vertex(pAlong + axisW * width * 0.5));\n        }\n        for (int s = 0; s < nSegs; ++s) {\n            mesh->add_triangle(leftVerts[s],   rightVerts[s],   rightVerts[s+1]);\n            mesh->add_triangle(leftVerts[s],   rightVerts[s+1], leftVerts[s+1]);\n        }\n    }\n    return true;\n}')
    assert old_func in content, 'ERROR: reconstruct_leaves not found!'
    content = content.replace(old_func, new_func, 1)
    print('  Patch B applied')
    old_bbox = ('\t//project the trunk points on the xy plane and get the bounding box\n\tdouble minX = DBL_MAX;\n\tdouble maxX = -DBL_MAX;\n\tdouble minY = DBL_MAX;\n\tdouble maxY = -DBL_MAX;\n\tfor (int nP = 0; nP < trunkList.size(); nP++)\n\t{\n\t\tif (minX > trunkList[nP].x)\n\t\t\tminX = trunkList[nP].x;\n\t\tif (maxX < trunkList[nP].x)\n\t\t\tmaxX = trunkList[nP].x;\n\t\tif (minY > trunkList[nP].y)\n\t\t\tminY = trunkList[nP].y;\n\t\tif (maxY < trunkList[nP].y)\n\t\t\tmaxY = trunkList[nP].y;\n\t}\n\n\t//assign the raw radius value and return\n    TrunkRadius_ = std::max((maxX - minX), (maxY - minY)) / 2.0;\n')
    new_bbox = ('\t// Patch D: Least-Squares circle fit instead of bounding box\n\tdouble minX = DBL_MAX, maxX = -DBL_MAX;\n\tdouble minY = DBL_MAX, maxY = -DBL_MAX;\n\tfor (int nP = 0; nP < (int)trunkList.size(); nP++) {\n\t\tif (minX > trunkList[nP].x) minX = trunkList[nP].x;\n\t\tif (maxX < trunkList[nP].x) maxX = trunkList[nP].x;\n\t\tif (minY > trunkList[nP].y) minY = trunkList[nP].y;\n\t\tif (maxY < trunkList[nP].y) maxY = trunkList[nP].y;\n\t}\n\tdouble cx = 0.0, cy = 0.0;\n\tfor (int nP = 0; nP < (int)trunkList.size(); nP++) {\n\t\tcx += trunkList[nP].x; cy += trunkList[nP].y;\n\t}\n\tcx /= (int)trunkList.size(); cy /= (int)trunkList.size();\n\tdouble r  = std::max((maxX - minX), (maxY - minY)) / 2.0;\n\tdouble r_max = r * 2.0;\n\tif ((int)trunkList.size() >= 1000) {\n\t\tfor (int iter = 0; iter < 100; ++iter) {\n\t\t\tdouble dCx=0, dCy=0, dR=0;\n\t\t\tdouble A00=0,A01=0,A02=0,A11=0,A12=0,A22=0,b0=0,b1=0,b2=0;\n\t\t\tfor (int nP = 0; nP < (int)trunkList.size(); nP++) {\n\t\t\t\tdouble dx=trunkList[nP].x-cx, dy=trunkList[nP].y-cy;\n\t\t\t\tdouble dist=std::sqrt(dx*dx+dy*dy);\n\t\t\t\tif (dist<1e-10) continue;\n\t\t\t\tdouble res=dist-r, Jcx=-dx/dist, Jcy=-dy/dist, Jr=-1.0;\n\t\t\t\tA00+=Jcx*Jcx; A01+=Jcx*Jcy; A02+=Jcx*Jr;\n\t\t\t\tA11+=Jcy*Jcy; A12+=Jcy*Jr;  A22+=Jr*Jr;\n\t\t\t\tb0+=Jcx*res;  b1+=Jcy*res;  b2+=Jr*res;\n\t\t\t}\n\t\t\tdouble det=A00*(A11*A22-A12*A12)-A01*(A01*A22-A12*A02)+A02*(A01*A12-A11*A02);\n\t\t\tif (std::abs(det)<1e-14) break;\n\t\t\tdCx=(b0*(A11*A22-A12*A12)-A01*(b1*A22-A12*b2)+A02*(b1*A12-A11*b2))/det;\n\t\t\tdCy=(A00*(b1*A22-A12*b2)-b0*(A01*A22-A12*A02)+A02*(A01*b2-b1*A02))/det;\n\t\t\tdR=(A00*(A11*b2-b1*A12)-A01*(A01*b2-b1*A02)+b0*(A01*A12-A11*A02))/det;\n\t\t\tcx-=dCx; cy-=dCy; r-=dR;\n\t\t\tif (r<0) r=std::abs(r);\n\t\t\tif (r>r_max) { r=r_max; break; }\n\t\t\tif (std::abs(dCx)+std::abs(dCy)+std::abs(dR)<1e-8) break;\n\t\t}\n\t}\n    TrunkRadius_ = r;\n')
    assert old_bbox in content, 'ERROR: BBox not found!'
    content = content.replace(old_bbox, new_bbox, 1)
    print('  Patch D applied')
    with open(skel, 'w') as f:
        f.write(content)
    print('  All patches applied')


def build_adtree():
    print('\nBuilding AdTree...')
    assert os.path.exists(ADTREE_SRC), f'AdTree-main not found at: {ADTREE_SRC}'
    for root, dirs, files in os.walk(ADTREE_SRC):
        for fname in files:
            if fname == 'CMakeLists.txt':
                fpath = os.path.join(root, fname)
                with open(fpath) as f:
                    c = f.read()
                if 'OpenGL::OpenGL' in c:
                    with open(fpath, 'w') as f:
                        f.write(c.replace('OpenGL::OpenGL', 'OpenGL::GL'))
    cmake_adtree = os.path.join(ADTREE_SRC, 'AdTree', 'CMakeLists.txt')
    with open(cmake_adtree) as f:
        c = f.read()
    c = c.replace('target_link_libraries(${PROJECT_NAME} PRIVATE easy3d_algo', 'target_link_libraries(${PROJECT_NAME} PRIVATE GLdispatch easy3d_algo')
    with open(cmake_adtree, 'w') as f:
        f.write(c)
    subprocess.run('ln -sf /usr/lib/x86_64-linux-gnu/libGLdispatch.so.0 /usr/lib/x86_64-linux-gnu/libGLdispatch.so 2>/dev/null || true', shell=True)
    print('Applying patches...')
    apply_patches(ADTREE_SRC)
    os.makedirs(ADTREE_BUILD, exist_ok=True)
    print('Running cmake...')
    subprocess.run(['cmake', '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS=-w', f'-DCMAKE_RUNTIME_OUTPUT_DIRECTORY={ADTREE_BUILD}/bin', ADTREE_SRC], cwd=ADTREE_BUILD, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ncpu = os.cpu_count() or 4
    print(f'Compiling with {ncpu} cores (3-5 min)...')
    subprocess.run(['make', f'-j{ncpu}'], cwd=ADTREE_BUILD, check=True)
    binary = None
    for root, dirs, files in os.walk(ADTREE_BUILD):
        for fname in files:
            fpath = os.path.join(root, fname)
            if fname == 'AdTree' and os.access(fpath, os.X_OK):
                binary = fpath
                break
    assert binary, 'ERROR: AdTree binary not found after compilation!'
    with open(ADTREE_BIN, 'w') as f:
        f.write(binary)
    print(f'AdTree compiled: {binary}')
    return binary


def filter_leaves_obj(leaves_in, leaves_out, keep_ratio, faces_per_leaf=12):
    with open(leaves_in) as f:
        lines = f.readlines()
    vertices = [l for l in lines if l.startswith('v ')]
    faces    = [l for l in lines if l.startswith('f ')]
    header   = [l for l in lines if not l.startswith('v ') and not l.startswith('f ')]
    n_leaves = len(faces) // faces_per_leaf
    n_keep   = max(1, int(n_leaves * keep_ratio))
    np.random.seed(42)
    keep_idx = np.sort(np.random.choice(n_leaves, n_keep, replace=False))
    kept_faces = []
    for idx in keep_idx:
        kept_faces.extend(faces[idx*faces_per_leaf : (idx+1)*faces_per_leaf])
    used_v = set()
    for face in kept_faces:
        for token in face.strip().split()[1:]:
            used_v.add(int(token.split('/')[0]) - 1)
    old_to_new = {}
    new_verts  = []
    for old_idx in sorted(used_v):
        old_to_new[old_idx] = len(new_verts) + 1
        new_verts.append(vertices[old_idx])
    new_faces = []
    for face in kept_faces:
        parts   = face.strip().split()
        new_idx = [str(old_to_new[int(p.split('/')[0]) - 1]) for p in parts[1:]]
        new_faces.append('f ' + ' '.join(new_idx) + '\n')
    with open(leaves_out, 'w') as f:
        f.writelines(header + new_verts + new_faces)


def translate_obj(obj_path, offset):
    with open(obj_path) as f:
        lines = f.readlines()
    result = []
    for line in lines:
        if line.startswith('v '):
            p = line.strip().split()
            result.append(f'v {float(p[1])+offset[0]:.6f} {float(p[2])+offset[1]:.6f} {float(p[3])+offset[2]:.6f}\n')
        else:
            result.append(line)
    return result


def merge_objs(obj_paths_and_offsets, out_path):
    all_lines, vertex_offset = [], 0
    for obj_path, offset in obj_paths_and_offsets:
        if not os.path.exists(obj_path):
            continue
        lines    = translate_obj(obj_path, offset)
        vertices = [l for l in lines if l.startswith('v ')]
        faces    = [l for l in lines if l.startswith('f ')]
        new_faces = []
        for face in faces:
            parts   = face.strip().split()
            new_idx = [str(int(p.split('/')[0]) + vertex_offset) for p in parts[1:]]
            new_faces.append('f ' + ' '.join(new_idx) + '\n')
        all_lines.extend(vertices + new_faces)
        vertex_offset += len(vertices)
    with open(out_path, 'w') as f:
        f.writelines(all_lines)
    print(f'  {os.path.basename(out_path):50s} {os.path.getsize(out_path)/1024/1024:.1f} MB')


def run_pointtree(input_file, scan_type, cloud_id):
    import pointtorch
    from pointtree.instance_segmentation import TreeXAlgorithm, TreeXPresetTLS, TreeXPresetULS
    print(f'\nLoading {input_file} ...')
    point_cloud = pointtorch.read(input_file)
    print(f'{len(point_cloud):,} points loaded')
    print(f'  Columns : {list(point_cloud.columns)}')
    print(f'  X range : {point_cloud["x"].min():.2f} - {point_cloud["x"].max():.2f}')
    print(f'  Y range : {point_cloud["y"].min():.2f} - {point_cloud["y"].max():.2f}')
    print(f'  Z range : {point_cloud["z"].min():.2f} - {point_cloud["z"].max():.2f}')
    preset = dict(TreeXPresetTLS() if scan_type == 'TLS' else TreeXPresetULS())
    preset['visualization_folder'] = None
    algorithm = TreeXAlgorithm(**preset)
    xyz = point_cloud[['x', 'y', 'z']].to_numpy()
    intensities = point_cloud['intensity'].to_numpy() if 'intensity' in point_cloud.columns else None
    print(f'Running TreeX ({scan_type}) on {len(xyz):,} points...')
    _out, _err = sys.stdout, sys.stderr
    class Quiet(io.RawIOBase):
        def write(self, b): return len(b) if isinstance(b, (bytes, str)) else 0
        def fileno(self): return sys.__stdout__.fileno()
    sys.stdout = sys.stderr = Quiet()
    try:
        instance_ids, trunk_positions, trunk_diameters = algorithm(xyz, intensities=intensities, point_cloud_id=cloud_id)
    finally:
        sys.stdout, sys.stderr = _out, _err
    tree_ids = np.unique(instance_ids); tree_ids = tree_ids[tree_ids >= 0]
    print(f'Trees detected: {len(tree_ids)}  |  Non-tree pts: {int(np.sum(instance_ids==-1)):,}')
    if len(trunk_diameters) > 0:
        print(f'Mean trunk diam: {np.mean(trunk_diameters):.3f} m')
    return point_cloud, xyz, instance_ids, tree_ids, trunk_positions, trunk_diameters


def save_pointtree_results(point_cloud, xyz, instance_ids, tree_ids, trunk_positions, trunk_diameters, cloud_id):
    import pandas as pd
    base_cols  = ['x', 'y', 'z', 'instance_id']
    extra_cols = [c for c in ['intensity', 'r', 'g', 'b'] if c in point_cloud.columns]
    save_cols  = base_cols + extra_cols
    point_cloud['instance_id'] = instance_ids
    labeled_path = os.path.join(POINTTREE_OUT, f'{cloud_id}_labeled.laz')
    point_cloud.to(labeled_path, columns=save_cols)
    print(f'Labeled cloud saved: {labeled_path}')
    trees_dir = os.path.join(POINTTREE_OUT, 'individual_trees')
    os.makedirs(trees_dir, exist_ok=True)
    save_cols_tree = [c for c in save_cols if c != 'instance_id']
    scene_origin = xyz.mean(axis=0)
    summary = []
    for i, tree_id in enumerate(tree_ids):
        mask       = instance_ids == tree_id
        tree_cloud = point_cloud[mask].copy()
        tree_xyz   = xyz[mask]
        bbox_min = tree_xyz.min(axis=0); bbox_max = tree_xyz.max(axis=0); centroid = tree_xyz.mean(axis=0)
        trunk_x = float(trunk_positions[i, 0]) if i < len(trunk_positions) else float(centroid[0])
        trunk_y = float(trunk_positions[i, 1]) if i < len(trunk_positions) else float(centroid[1])
        trunk_z = float(bbox_min[2])
        diam    = float(trunk_diameters[i]) if i < len(trunk_diameters) else None
        out_path = os.path.join(trees_dir, f'tree_{tree_id:04d}.laz')
        tree_cloud.to(out_path, columns=save_cols_tree)
        meta = {'tree_id': int(tree_id), 'file': f'tree_{tree_id:04d}.laz', 'n_points': int(mask.sum()), 'trunk_x': round(trunk_x,4), 'trunk_y': round(trunk_y,4), 'trunk_z': round(trunk_z,4), 'trunk_diam_m': round(diam,4) if diam else None, 'centroid_x': round(float(centroid[0]),4), 'centroid_y': round(float(centroid[1]),4), 'centroid_z': round(float(centroid[2]),4), 'bbox_min': [round(float(v),4) for v in bbox_min], 'bbox_max': [round(float(v),4) for v in bbox_max], 'height_m': round(float(bbox_max[2]-bbox_min[2]),3), 'offset_from_scene_origin': [round(float(trunk_x-scene_origin[0]),4), round(float(trunk_y-scene_origin[1]),4), round(float(trunk_z-scene_origin[2]),4)]}
        summary.append(meta)
        with open(os.path.join(trees_dir, f'tree_{tree_id:04d}_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
    assembly = {'cloud_id': cloud_id, 'scene_origin': [round(float(v),4) for v in scene_origin], 'n_trees': len(tree_ids), 'trees': summary}
    assembly_path = os.path.join(POINTTREE_OUT, f'{cloud_id}_assembly.json')
    with open(assembly_path, 'w') as f:
        json.dump(assembly, f, indent=2)
    df = pd.DataFrame(summary)
    csv_path = os.path.join(POINTTREE_OUT, f'{cloud_id}_tree_summary.csv')
    df.to_csv(csv_path, index=False)
    print(f'{len(tree_ids)} trees saved: {trees_dir}')
    print(f'Assembly JSON: {assembly_path}  |  Summary CSV: {csv_path}')
    print(df[['tree_id','n_points','height_m','trunk_diam_m','trunk_x','trunk_y']].to_string(index=False))
    return summary, trees_dir


def run_adtree(summary, trees_dir, adtree_bin):
    import laspy, open3d as o3d
    print(f'\nAdTree binary: {adtree_bin}')
    for tree in summary:
        tid = f'tree_{tree["tree_id"]:04d}'
        laz_path = os.path.join(trees_dir, tree['file'])
        tree_out = os.path.join(ADTREE_OUT, tid)
        os.makedirs(tree_out, exist_ok=True)
        print(f'\n{tid}  ({tree["n_points"]:,} pts  H={tree["height_m"]:.1f}m)')
        las = laspy.read(laz_path)
        pts = np.vstack([las.x, las.y, las.z]).T
        print(f'  {len(pts):,} points loaded')
        if ENABLE_NOISE_FILTER:
            pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
            pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
            pts = np.asarray(pcd.points); print(f'  After filtering: {len(pts):,} points')
        pcd_save = o3d.geometry.PointCloud(); pcd_save.points = o3d.utility.Vector3dVector(pts)
        o3d.io.write_point_cloud(os.path.join(tree_out, f'{tid}_pointcloud.ply'), pcd_save)
        xyz_file = os.path.join(tree_out, f'{tid}.xyz')
        np.savetxt(xyz_file, pts, fmt='%.6f')
        adtree_raw = os.path.join(tree_out, 'adtree_raw')
        os.makedirs(adtree_raw, exist_ok=True)
        proc = subprocess.run([adtree_bin, xyz_file, adtree_raw, '-skeleton'], capture_output=True, text=True, timeout=900)
        src_br = os.path.join(adtree_raw, f'{tid}_branches.obj')
        src_lv = os.path.join(adtree_raw, f'{tid}_leaves.obj')
        src_sk = os.path.join(adtree_raw, f'{tid}_skeleton.ply')
        if not os.path.exists(src_br):
            print(f'  ERROR: AdTree failed - skipping'); print(proc.stderr[-300:]); tree['adtree_success'] = False; continue
        tree['adtree_success'] = True; tree['output_dir'] = tree_out
        shutil.copy(src_br, os.path.join(tree_out, f'{tid}_branches.obj'))
        shutil.copy(src_sk, os.path.join(tree_out, f'{tid}_skeleton.ply'))
        leaves_orig = os.path.join(tree_out, f'{tid}_leaves.obj')
        shutil.copy(src_lv, leaves_orig)
        for ratio in KEEP_RATIOS:
            ratio_str = str(ratio).replace('.', '')
            filter_leaves_obj(leaves_orig, os.path.join(tree_out, f'{tid}_leaves_filtered{ratio_str}.obj'), ratio, FACES_PER_LEAF)
        print(f'  Done -> {tree_out}')
    n_ok = sum(1 for t in summary if t.get('adtree_success', False))
    print(f'\nAdTree complete: {n_ok}/{len(summary)} trees successful')
    return summary


def merge_forest(summary):
    import open3d as o3d
    forest_out = os.path.join(ADTREE_OUT, 'forest')
    os.makedirs(forest_out, exist_ok=True)
    success_trees = [t for t in summary if t.get('adtree_success', False)]
    print(f'\nMerging {len(success_trees)} trees...')
    def get_offset(t): return t.get('offset_from_scene_origin', [0,0,0])
    all_pts = []
    for t in success_trees:
        ply = os.path.join(t['output_dir'], f'tree_{t["tree_id"]:04d}_pointcloud.ply')
        if os.path.exists(ply):
            pcd = o3d.io.read_point_cloud(ply); all_pts.append(np.asarray(pcd.points) + np.array(get_offset(t)))
    if all_pts:
        merged = np.vstack(all_pts); pcd_f = o3d.geometry.PointCloud(); pcd_f.points = o3d.utility.Vector3dVector(merged)
        o3d.io.write_point_cloud(os.path.join(forest_out, 'forest_pointcloud.ply'), pcd_f)
        print(f'  forest_pointcloud.ply  ({len(merged):,} points)')
    merge_objs([(os.path.join(t['output_dir'], f'tree_{t["tree_id"]:04d}_branches.obj'), get_offset(t)) for t in success_trees], os.path.join(forest_out, 'forest_branches.obj'))
    merge_objs([(os.path.join(t['output_dir'], f'tree_{t["tree_id"]:04d}_leaves.obj'), get_offset(t)) for t in success_trees], os.path.join(forest_out, 'forest_leaves.obj'))
    for ratio in KEEP_RATIOS:
        ratio_str = str(ratio).replace('.', '')
        merge_objs([(os.path.join(t['output_dir'], f'tree_{t["tree_id"]:04d}_leaves_filtered{ratio_str}.obj'), get_offset(t)) for t in success_trees], os.path.join(forest_out, f'forest_leaves_filtered{ratio_str}.obj'))
    print('Forest merge complete.')


def create_zip(cloud_id):
    import zipfile
    final_zip = os.path.join(BASE_DIR, 'output', f'{cloud_id}_results.zip')
    print(f'\nCreating ZIP: {final_zip}')
    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for out_dir, prefix in [(POINTTREE_OUT, 'pointtree'), (ADTREE_OUT, 'adtree')]:
            for root, dirs, files in os.walk(out_dir):
                for fname in files:
                    full = os.path.join(root, fname)
                    zf.write(full, os.path.join(prefix, os.path.relpath(full, out_dir)))
    print(f'ZIP created: {final_zip}  ({os.path.getsize(final_zip)/1024/1024:.1f} MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Forest Reconstruction Pipeline')
    parser.add_argument('--build', action='store_true', help='Compile AdTree (run once)')
    parser.add_argument('--input', type=str, help='Input point cloud file')
    parser.add_argument('--scan-type', type=str, default='TLS', choices=['TLS','ULS'], help='TLS or ULS')
    args = parser.parse_args()

    if args.build:
        build_adtree(); sys.exit(0)

    if args.input:
        input_file = args.input
    else:
        found = []
        for ext in ['*.las','*.laz','*.ply','*.txt','*.csv']:
            found += glob.glob(os.path.join(BASE_DIR, 'input', ext))
        assert found, 'No input file found in input/ folder. Use --input path/to/file.laz'
        input_file = found[0]
        if len(found) > 1: print(f'Multiple files found - using: {os.path.basename(input_file)}')

    assert os.path.exists(input_file), f'Input file not found: {input_file}'
    assert os.path.exists(ADTREE_BIN), 'AdTree not compiled. Run: python pipeline.py --build'
    with open(ADTREE_BIN) as f: adtree_bin = f.read().strip()
    assert os.path.exists(adtree_bin), f'AdTree binary not found: {adtree_bin}'

    cloud_id = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(POINTTREE_OUT, exist_ok=True)
    os.makedirs(ADTREE_OUT, exist_ok=True)

    print(f'Input file : {input_file}')
    print(f'Scan type  : {args.scan_type}')
    print(f'Cloud ID   : {cloud_id}')

    t0 = time.time()
    point_cloud, xyz, instance_ids, tree_ids, trunk_positions, trunk_diameters = run_pointtree(input_file, args.scan_type, cloud_id)
    summary, trees_dir = save_pointtree_results(point_cloud, xyz, instance_ids, tree_ids, trunk_positions, trunk_diameters, cloud_id)
    summary = run_adtree(summary, trees_dir, adtree_bin)
    merge_forest(summary)
    create_zip(cloud_id)
    print(f'\nTotal time: {time.time()-t0:.0f}s')
    print(f'Output: {os.path.join(BASE_DIR, "output")}')
