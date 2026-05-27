#!/usr/bin/env python3
"""
Single Tree Reconstruction Pipeline
=====================================
AdTree-based reconstruction for a single tree point cloud.

Usage:
    python single_tree.py --build                   # compile AdTree once
    python single_tree.py --input tree.laz
    python single_tree.py --input tree.laz --no-filter
"""

import os, sys, json, shutil, subprocess, argparse, time, glob
import numpy as np

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ADTREE_SRC   = os.path.join(BASE_DIR, 'AdTree-main')
ADTREE_BUILD = os.path.join(BASE_DIR, 'AdTree-build')
ADTREE_BIN   = os.path.join(BASE_DIR, 'adtree_path.txt')
OUTPUT_DIR   = os.path.join(BASE_DIR, 'output')

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
                with open(fpath) as f: c = f.read()
                if 'OpenGL::OpenGL' in c:
                    with open(fpath, 'w') as f: f.write(c.replace('OpenGL::OpenGL', 'OpenGL::GL'))
    cmake_adtree = os.path.join(ADTREE_SRC, 'AdTree', 'CMakeLists.txt')
    with open(cmake_adtree) as f: c = f.read()
    c = c.replace('target_link_libraries(${PROJECT_NAME} PRIVATE easy3d_algo', 'target_link_libraries(${PROJECT_NAME} PRIVATE GLdispatch easy3d_algo')
    with open(cmake_adtree, 'w') as f: f.write(c)
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
            if fname == 'AdTree' and os.access(fpath, os.X_OK): binary = fpath; break
    assert binary, 'ERROR: AdTree binary not found!'
    with open(ADTREE_BIN, 'w') as f: f.write(binary)
    print(f'AdTree compiled: {binary}')
    return binary


def filter_leaves_obj(leaves_in, leaves_out, keep_ratio, faces_per_leaf=12):
    with open(leaves_in) as f: lines = f.readlines()
    vertices = [l for l in lines if l.startswith('v ')]
    faces    = [l for l in lines if l.startswith('f ')]
    header   = [l for l in lines if not l.startswith('v ') and not l.startswith('f ')]
    n_leaves = len(faces) // faces_per_leaf
    n_keep   = max(1, int(n_leaves * keep_ratio))
    np.random.seed(42)
    keep_idx = np.sort(np.random.choice(n_leaves, n_keep, replace=False))
    kept_faces = []
    for idx in keep_idx: kept_faces.extend(faces[idx*faces_per_leaf:(idx+1)*faces_per_leaf])
    used_v = set()
    for face in kept_faces:
        for token in face.strip().split()[1:]: used_v.add(int(token.split('/')[0]) - 1)
    old_to_new = {}; new_verts = []
    for old_idx in sorted(used_v): old_to_new[old_idx] = len(new_verts) + 1; new_verts.append(vertices[old_idx])
    new_faces = []
    for face in kept_faces:
        parts = face.strip().split()
        new_idx = [str(old_to_new[int(p.split('/')[0]) - 1]) for p in parts[1:]]
        new_faces.append('f ' + ' '.join(new_idx) + '\n')
    with open(leaves_out, 'w') as f: f.writelines(header + new_verts + new_faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Single Tree Reconstruction Pipeline')
    parser.add_argument('--build', action='store_true', help='Compile AdTree (run once)')
    parser.add_argument('--input', type=str, help='Input point cloud file (.las .laz .ply .txt .csv)')
    parser.add_argument('--no-filter', action='store_true', help='Disable noise filtering even if ENABLE_NOISE_FILTER=True')
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
    assert os.path.exists(ADTREE_BIN), 'AdTree not compiled. Run: python single_tree.py --build'
    with open(ADTREE_BIN) as f: adtree_bin = f.read().strip()
    assert os.path.exists(adtree_bin), f'AdTree binary not found: {adtree_bin}'

    tree_id = os.path.splitext(os.path.basename(input_file))[0]
    tree_out = os.path.join(OUTPUT_DIR, tree_id)
    os.makedirs(tree_out, exist_ok=True)

    print(f'Input file : {input_file}')
    print(f'Tree ID    : {tree_id}')
    print(f'Output dir : {tree_out}')

    t0 = time.time()

    # Load point cloud
    import laspy, open3d as o3d
    ext = os.path.splitext(input_file)[1].lower()
    if ext in ('.las', '.laz'):
        las = laspy.read(input_file); pts = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
    elif ext == '.ply':
        pcd = o3d.io.read_point_cloud(input_file); pts = np.asarray(pcd.points).astype(np.float64)
    elif ext in ('.txt', '.csv'):
        pts = np.loadtxt(input_file, delimiter=None)[:, :3].astype(np.float64)
    else:
        raise ValueError(f'Unsupported format: {ext}')

    print(f'{len(pts):,} points loaded')
    print(f'  X range : {pts[:,0].min():.2f} - {pts[:,0].max():.2f}')
    print(f'  Y range : {pts[:,1].min():.2f} - {pts[:,1].max():.2f}')
    print(f'  Z range : {pts[:,2].min():.2f} - {pts[:,2].max():.2f}')

    z_range  = pts[:,2].max() - pts[:,2].min()
    xy_range = max(pts[:,0].max()-pts[:,0].min(), pts[:,1].max()-pts[:,1].min())
    if z_range < xy_range * 0.3:
        print('WARNING: Z range is very small relative to XY - check that Z points upward')

    # Optional noise filtering
    if ENABLE_NOISE_FILTER and not args.no_filter:
        pcd = o3d.geometry.PointCloud(); pcd.points = o3d.utility.Vector3dVector(pts)
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
        pcd = pcd.voxel_down_sample(voxel_size=VOXEL_SIZE)
        pts = np.asarray(pcd.points).astype(np.float64)
        print(f'After filtering: {len(pts):,} points')
    else:
        print('Noise filtering skipped.')

    # Save PLY
    pcd_save = o3d.geometry.PointCloud(); pcd_save.points = o3d.utility.Vector3dVector(pts)
    ply_path = os.path.join(tree_out, f'{tree_id}_pointcloud.ply')
    o3d.io.write_point_cloud(ply_path, pcd_save)
    print(f'Point cloud saved: {ply_path}')

    # Write XYZ for AdTree
    xyz_file = os.path.join(tree_out, f'{tree_id}.xyz')
    np.savetxt(xyz_file, pts, fmt='%.6f')

    # Run AdTree
    adtree_raw = os.path.join(tree_out, 'adtree_raw')
    os.makedirs(adtree_raw, exist_ok=True)
    print(f'Running AdTree on {len(pts):,} points...')
    proc = subprocess.run([adtree_bin, xyz_file, adtree_raw, '-skeleton'], capture_output=True, text=True, timeout=900)

    src_br = os.path.join(adtree_raw, f'{tree_id}_branches.obj')
    src_lv = os.path.join(adtree_raw, f'{tree_id}_leaves.obj')
    src_sk = os.path.join(adtree_raw, f'{tree_id}_skeleton.ply')

    assert os.path.exists(src_br), f'AdTree failed.\nstderr: {proc.stderr[-500:]}'

    shutil.copy(src_br, os.path.join(tree_out, f'{tree_id}_branches.obj'))
    shutil.copy(src_sk, os.path.join(tree_out, f'{tree_id}_skeleton.ply'))
    leaves_orig = os.path.join(tree_out, f'{tree_id}_leaves.obj')
    shutil.copy(src_lv, leaves_orig)
    print(f'Branches : {tree_out}/{tree_id}_branches.obj')
    print(f'Leaves   : {tree_out}/{tree_id}_leaves.obj')
    print(f'Skeleton : {tree_out}/{tree_id}_skeleton.ply')

    # Filter leaves
    for ratio in KEEP_RATIOS:
        ratio_str = str(ratio).replace('.', '')
        out_path  = os.path.join(tree_out, f'{tree_id}_leaves_filtered{ratio_str}.obj')
        filter_leaves_obj(leaves_orig, out_path, ratio, FACES_PER_LEAF)
        print(f'Leaves filtered {int(ratio*100)}% : {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)')

    # Create ZIP
    import zipfile
    final_zip = os.path.join(OUTPUT_DIR, f'{tree_id}_results.zip')
    print(f'\nCreating ZIP: {final_zip}')
    with zipfile.ZipFile(final_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(tree_out):
            for fname in files:
                full = os.path.join(root, fname)
                zf.write(full, os.path.relpath(full, tree_out))
    print(f'ZIP created: {final_zip}  ({os.path.getsize(final_zip)/1024/1024:.1f} MB)')
    print(f'\nTotal time: {time.time()-t0:.0f}s')
    print(f'Output: {tree_out}')
