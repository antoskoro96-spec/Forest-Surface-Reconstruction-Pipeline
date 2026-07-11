# Forest-Surface-Reconstruction-Pipeline 

Surface reconstruction from forest point clouds

A combined pipeline for forest point cloud segmentation and individual tree reconstruction, built on top of [PointTree](https://github.com/ai4trees/pointtree) and [AdTree](https://github.com/tudelft3d/AdTree). [TreeNet3D](https://github.com/ao216/TreeNet3D) is used as a source of ground truth single-tree data for validation and testing of the AdTree reconstruction.

---

## Credits

This project builds directly on the following open-source works. Please cite and credit these projects if you use this pipeline.

| Project | Authors | Repository |
|---|---|---|
| **AdTree** | Shenglan Du, Roderik Lindenbergh, Hugo Ledoux, Jantien Stoter, Liangliang Nan | [tudelft3d/AdTree](https://github.com/tudelft3d/AdTree) |
| **PointTree** | Jan Windheuser et al. | [ai4trees/pointtree](https://github.com/ai4trees/pointtree) |

> **Note on validation data:** Individual tree point clouds for testing were obtained from the [TreeNet3D dataset](https://github.com/ao216/TreeNet3D) (Tang et al., 2024). TreeNet3D data requires a signed agreement with the authors and cannot be redistributed. Contact shengjuntang@szu.edu.cn with your institutional email to request access.

> AdTree is described in: *Shenglan Du, Roderik Lindenbergh, Hugo Ledoux, Jantien Stoter, and Liangliang Nan. AdTree: Accurate, Detailed, and Automatic Modelling of Laser-Scanned Trees. Remote Sensing, 11(18), 2074, 2019. https://doi.org/10.3390/rs11182074*

---

## What This Project Does

### Forest Pipeline (`pipeline.py` / `forest_reconstruction_pipeline.ipynb`)

Takes a segmented forest point cloud as input and runs a two-stage pipeline:

1. **PointTree** segments the full forest scan into individual tree instances, detects trunk positions and estimates trunk diameters.
2. **AdTree** reconstructs each individual tree — generating a skeleton, branch mesh and leaf mesh.

The outputs are merged back into a complete forest and packaged as a ZIP.




| <img src="https://github.com/user-attachments/assets/7ae4b63a-79d4-4911-bf63-38eb9e514418" alt="Point Cloud" width="220" height="500"> | <img src="https://github.com/user-attachments/assets/224744a4-bc23-4c1a-9ae1-b65299907390" alt="Tree instance segmentation" width="220" height="500"> | <img src="https://github.com/user-attachments/assets/9c7799b4-d01d-442f-9091-ba6f644e164b" alt="AdTree wood mesh" width="220" height="500"> | <img src="https://github.com/user-attachments/assets/87c7a51c-75fd-4d2d-b651-209fbd101188" alt="AdTree leaf mesh" width="220" height="500"> |
| :---: | :---: | :---: | :---: |
| Point Cloud | Tree instance segmentation | AdTree wood mesh | AdTree leaf mesh |

### Single Tree Pipeline (`single_tree.py` / `single_tree_reconstruction.ipynb`)

Takes a single-tree point cloud as input and runs AdTree directly. Designed for testing and validation of individual trees. Accepts any `.las`, `.laz`, `.ply`, `.txt` or `.csv` file.

---

## Changes Made to AdTree

The original AdTree C++ source is modified with nine patches applied automatically before compilation. The patches are grouped and ordered to mirror the three reconstruction stages described in the paper — **skeleton extraction**, **wood-mesh reconstruction**, and **leaf generation** — and are numbered consecutively (1–9) across the groups. All patches are applied automatically by the pipeline scripts.


### 1. Skeleton Extraction

| <img src="https://github.com/user-attachments/assets/01632bac-cf79-4df5-9d94-a955315e9ad3" alt="skeleton_old" width="290"> | <img src="https://github.com/user-attachments/assets/0f1fcb3c-4faa-4c1d-962f-7540d6c4fdd3" alt="skeleton_new" width="290"> |
| :---: | :---: |
| Before | After |

#### Patch 1 — Initial trunk-point range increased from 2% to 10%

**Before:** Only points within the lowest 2% of the tree height were used to estimate the initial trunk radius.

```cpp
double epsiony = 0.02;
```

**After:** The threshold is increased to the lowest 10% of the tree height.

```cpp
double epsiony = 0.10;
```

**Why:** The initial trunk radius is used as a scale parameter during main-branch point centralization. Using a larger lower-trunk region provides more points and can make this initial estimate more stable, especially for sparse or noisy point clouds. This radius is not necessarily the final mesh radius; the final radius is recalibrated later in Patch 6.

---

#### Patch 2 — Improved initial trunk radius estimate by least-squares circle fitting

**Before:** The initial trunk radius was estimated from the 2D bounding box of the selected lower trunk points projected onto the XY plane.

```cpp
TrunkRadius_ = std::max((maxX - minX), (maxY - minY)) / 2.0;
```

This estimate can become too large when the selected lower region contains outliers, nearby branch points, or elongated point distributions.

**After:** The bounding-box estimate is replaced by a 2D Gauss-Newton least-squares circle fit. The fit estimates a circle center and radius from the selected lower trunk points. The resulting radius is used as the initial `TrunkRadius_` for the following skeleton centralization step. If too few trunk points are available, the original bounding-box estimate remains the fallback.

**Why:** The initial trunk radius controls the neighborhood size used during main-branch point centralization. A more stable initial radius can improve the extracted skeleton by reducing under- or over-centralization. The fitted circle center is only used internally for estimating the radius; the skeleton later uses the radius value, not the fitted center. The final mesh radius is recalibrated separately in Patch 6.

---

#### Patch 3 — Adaptive skeleton simplification

**Before:** A single fixed merge threshold was used for the whole tree when simplifying the skeleton. A vertex was merged whenever its deviation was below `1.0 * r`. This could simplify the trunk and main-branch regions too aggressively and remove important curvature.

```cpp
double r = (*i_Graph)[edge(i_dVertex, parentV, *i_Graph).first].nRadius;
if (distance >= 1.0 * r)
    return false;
```

**After:** The merge threshold becomes dependent on the relative subtree length:

`fraction = lengthOfSubtree(node) / lengthOfSubtree(root)`

Large `fraction` values correspond to large subtrees, typically trunk and main-branch regions, while small values correspond to terminal branches. A smoothstep transition over the band `[0.05, 0.25]` reduces the merge threshold for large subtrees, making merging stricter and preserving curvature. Terminal branches keep the original, more aggressive simplification behavior.

```cpp
double fraction = (rootSubtree > 1e-10) ? (nodeSubtree / rootSubtree) : 0.0;
double bandLo = 0.05, bandHi = 0.25, curveStrength = 0.1;
double tt = clamp((fraction - bandLo) / (bandHi - bandLo), 0.0, 1.0);
double s  = tt*tt*(3.0 - 2.0*tt);                 // smoothstep
double mergeThreshold = 1.0 + (curveStrength - 1.0) * s;
if (distance >= mergeThreshold * r)
    return false;
```

**Why:** The uniform simplification could oversimplify the main stem and main branches. The adaptive rule preserves curvature in structurally important regions while still removing redundant detail in terminal crown branches, as described in Section III-A of the paper.

---

#### Patch 4 — Junction-aware skeleton smoothing and gap filling

**Before:** The reconstructed centerline and radii were taken directly from the cubic interpolation of each branch path. This could produce wavy centerlines, jittery radius profiles, and long straight jumps where the interpolated points were sparse.

**After:** Each branch path is tagged with `hardAnchors`, marking root points, branch junctions, and branch tips that must not be moved. The interpolated path is then post-processed:

* **Centerline smoothing:** 4 passes of windowed averaging that never crosses a hard anchor.
* **Radius smoothing:** 8 smoothing passes along the path, while preserving hard anchors.
* **Tip handling:** closed tips are temporarily protected during radius smoothing and then restored to a zero radius.
* **Monotonic radius cleanup:** branch radii are constrained to not increase toward the tip.
* **Straight-gap filling:** additional points are inserted where the spacing between consecutive centerline samples exceeds `1.4 ×` the typical spacing.

Hard anchors remain fixed throughout this cleanup, so roots, branch junctions, and tips are preserved.

**Why:** Smoother centerlines and radius profiles produce more continuous branch cylinders. Gap filling avoids long cylinder shortcuts caused by sparse interpolation, while hard anchors prevent the smoothing step from moving important topological points.

---

#### Patch 5 — Lower trunk straightening

**Before:** The lowest section of the main trunk followed the raw reconstructed skeleton. Near the base, this could cause visible wobbling or sideways offsets due to root, ground, or scan artifacts.

**After:** On the main path only, the lowest approximately 5% of the trunk is straightened. An attachment point is selected at `5% × TreeHeight`, a local trunk direction is estimated from the following points above it, and the points below are projected onto this line while keeping their original height values.

**Why:** The lower trunk is visually important and often affected by reconstruction artifacts. Straightening only the lowest main-trunk section reduces base wobbling without changing the overall tree topology or branch structure.

---

### 2. Wood-Mesh Reconstruction

| <img width="260" height="570" alt="front3d" src="https://github.com/user-attachments/assets/285692cd-15e9-4701-92d6-55658d4f75a9" /> | <img width="260" height="570" alt="front3" src="https://github.com/user-attachments/assets/985bc2e1-acee-4223-a585-2005781c76fc" /> |
| :---: | :---: |
| Before | After |

#### Patch 6 — Final trunk radius calibration

**Before:** The current `TrunkRadius_` was passed directly to `compute_all_edges_radius(TrunkRadius_)`, which propagates the trunk radius to the remaining branches. If this value was too large, the entire wood mesh became too thick.

```cpp
compute_all_edges_radius(TrunkRadius_);
```

**After:** Just before branch-radius propagation, `TrunkRadius_` is recalibrated from the reconstructed tree structure. The code iterates over all edges of the simplified skeleton, collects the original input points assigned to these edges through `vecPoints`, and keeps only those located in the lowest 2% of the tree height. These filtered lower points are projected onto the XY plane. Their centroid is computed, and the final trunk radius is set to the median radial distance from the centroid.

**Why:** This separates the radius used for early skeleton centralization from the radius used for final wood-mesh thickness. Patch 2 provides an initial radius for skeleton extraction, while Patch 6 recalibrates the final radius directly before it is propagated to the branch radii. Using points assigned to the simplified skeleton provides a filtered point set and makes the final radius less sensitive to raw point-cloud outliers.

---

### 3. Leaf Generation

| <img width="260" height="570" alt="front4d" src="https://github.com/user-attachments/assets/0b607f42-c42e-42c4-b2fa-82f46dd2c803" /> | <img width="260" height="570" alt="front4" src="https://github.com/user-attachments/assets/b296f416-4eeb-4609-bb5f-f12103b5d166" /> |
| :---: | :---: |
| Before | After |

#### Patch 7 — Leaf density and size reduction

**Before:** Each terminal vertex could generate up to 10 leaves. The leaf size and scatter radius were also relatively large, producing very dense canopies that often obscured the reconstructed wood mesh.

```cpp
int density = ceil(random_float() * 10);
generate_leaves(currentLeafVertex, 0.05);
double radius = 0.2 / log((float)num_edges(simplified_skeleton_));
```

**After:** Leaf density is reduced to approximately one leaf per terminal vertex, while leaf size and scatter radius are also reduced.

```cpp
int density = ceil(random_float() * 1);
generate_leaves(currentLeafVertex, 0.02);
double radius = 0.04 / log((float)num_edges(simplified_skeleton_));
```

**Why:** The original settings produced oversized and overly dense leaf meshes. Reducing density, size, and scatter radius makes the leaf mesh lighter, less visually cluttered, and allows the reconstructed wood structure to remain visible.

---

#### Patch 8 — Leaf base attached to branch tip

**Before:** Leaf positions were scattered around a point located along the direction from the branch tip toward its parent. Leaf directions were chosen randomly, so leaves could float away from the branch or point in unnatural directions.

```cpp
vec3 pEnd = pCurrent - (random_float() / 2.0) * ((pCurrent - pParent).normalize());
vec3 dirLeaf = random_direction();
vec3 pLeaf = pEnd + dirLeaf * random_float() * radius;
```

**After:** The leaf base is placed close to the terminal branch tip with only a small offset along the branch direction. The leaf direction is generated from a perpendicular component blended with the branch direction, creating an outward and slightly forward orientation.

```cpp
vec3 branchDir = (pCurrent - pParent).normalize();
double offset = random_float() * radius * 0.5;
vec3 pLeaf = pCurrent - branchDir * offset;
vec3 dirLeaf = (randPerp * 0.6f + branchDir * 0.4f).normalize();
```

**Why:** Leaves should visually grow from branch tips rather than float around them. This patch anchors the leaf base closer to the terminal branch and gives the leaf a more consistent orientation relative to the branch direction.

**Implementation note:** The perpendicular random direction should be normalized safely. If the random vector is nearly parallel to the branch direction, a fallback perpendicular vector should be used before normalization.

---

#### Patch 9 — Elliptic leaf shape

**Before:** Each leaf was represented as a flat rectangular quad made of two triangles.

**After:** Each leaf is constructed as a segmented elliptic strip with 6 segments and a sine-based width profile. The width is small at the base and tip and wider near the middle.

```cpp
const int nSegs = 6;
for (int s = 0; s <= nSegs; ++s) {
    double t = (double)s / nSegs;
    double width = sin(M_PI * t) * (2.0 - 0.3 * t);  // tapered elliptic profile
    ...
}
```

**Why:** Rectangular leaf quads look artificial. The segmented elliptic profile creates a more natural leaf shape with a tapered base, wider middle section, and pointed tip.

---

## Requirements

### System dependencies (Linux / WSL2)
```
cmake, build-essential, libboost-all-dev
libgl1-mesa-dev, libglu1-mesa-dev
libxrandr-dev, libxinerama-dev, libxcursor-dev, libxi-dev, libxext-dev
```

### System dependencies (macOS)
```
cmake, boost  (via Homebrew)
```

### Python packages — Forest Pipeline
```
numpy==1.26.4
torch==2.5.0 (CPU)
torch-scatter, torch-cluster
pointtree, pointtorch
laspy[lazrs], open3d, scipy, pandas
```

### Python packages — Single Tree Pipeline
```
numpy, laspy[lazrs], open3d
```

---

## Folder Structure

```
project/
    AdTree-main/              <- unzipped AdTree source (required)
    input/                    <- place your input file here
    output/                   <- created automatically
    pipeline.py               <- forest pipeline script
    single_tree.py            <- single tree script
    forest_reconstruction_pipeline.ipynb
    single_tree_reconstruction.ipynb
    README.md
```

`AdTree-main/` must be the **unzipped folder**, not the zip file.
Download it from [tudelft3d/AdTree](https://github.com/tudelft3d/AdTree).

---

## Usage

### Option 1 — Google Colab

Upload `Forest_Reconstruction_Pipeline.ipynb` or `single_tree_reconstruction.ipynb` to [colab.research.google.com](https://colab.research.google.com).

Upload your input file and `AdTree-main.zip` to the Colab session.
Then run all cells in order.

- **Step 1** installs Python packages (kernel restarts automatically afterwards)
- **Step 2** installs system dependencies and compiles AdTree (3–5 min)
- **Steps 3+** run the pipeline
- The final step downloads a ZIP of all results

> Colab provides a free Linux environment with no local setup required.
> GPU is not needed — the pipeline runs on CPU.

---

### Option 2 — Windows (WSL2)

WSL2 gives you a full Ubuntu environment inside Windows.

**One-time setup:**

```powershell
# In PowerShell as Administrator
wsl --install
```
Restart. Ubuntu opens and asks for a username/password.

```bash
# In Ubuntu terminal
sudo apt-get update
sudo apt-get install -y cmake build-essential libboost-all-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc

# Create environment
conda create -n adtree python=3.10 -y
conda activate adtree

# Install packages (forest pipeline)
pip install numpy==1.26.4
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install pointtree pointtorch laspy[lazrs] open3d scipy pandas

# Copy project to Linux filesystem (better performance than /mnt/c/)
cp -r /mnt/c/Users/YourName/project ~/project
cd ~/project

# Compile AdTree (once, 3-5 min)
python pipeline.py --build
```

**Running:**
```bash
conda activate adtree
cd ~/project
python pipeline.py --input input/forest.laz
python pipeline.py --input input/forest.laz --scan-type ULS
python single_tree.py --input input/tree.laz
```

Output is at `~/project/output/`. Access from Windows Explorer at:
`\\wsl$\Ubuntu\home\YourName\project\output\`

---

### Option 3 — macOS

```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# System dependencies
brew install cmake boost

# Install Miniconda (Intel)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3

# Install Miniconda (Apple Silicon M1/M2/M3)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh
bash Miniconda3-latest-MacOSX-arm64.sh -b -p $HOME/miniconda3

eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.zshrc

# Create environment and install packages
conda create -n adtree python=3.10 -y
conda activate adtree
pip install numpy==1.26.4
pip install torch==2.5.0
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install pointtree pointtorch laspy[lazrs] open3d scipy pandas

# Compile AdTree (once)
cd ~/project
python pipeline.py --build
```

> **Note for macOS:** if cmake fails with an OpenGL error, open `pipeline.py`,
> find `build_adtree()`, and remove the two lines that patch `GLdispatch` into
> the CMakeLists — this fix is only needed on Linux.

**Running:**
```bash
conda activate adtree
cd ~/project
python pipeline.py --input input/forest.laz
python single_tree.py --input input/tree.laz
```

---

### Option 4 — Linux

```bash
# System dependencies
sudo apt-get update
sudo apt-get install -y cmake build-essential libboost-all-dev \
    libgl1-mesa-dev libglu1-mesa-dev \
    libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev libxext-dev

# Install Miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
echo 'eval "$($HOME/miniconda3/bin/conda shell.bash hook)"' >> ~/.bashrc
source ~/.bashrc

# Create environment and install packages
conda create -n adtree python=3.10 -y
conda activate adtree
pip install numpy==1.26.4
pip install torch==2.5.0 --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install pointtree pointtorch laspy[lazrs] open3d scipy pandas

# Compile AdTree (once)
cd ~/project
python pipeline.py --build
```

**Running:**
```bash
conda activate adtree
cd ~/project
python pipeline.py --input input/forest.laz
python single_tree.py --input input/tree.laz
```

---

## Command Reference

### pipeline.py

| Command | Description |
|---|---|
| `python pipeline.py --build` | Compile AdTree — run this once before first use |
| `python pipeline.py` | Auto-detect input file from `input/` folder |
| `python pipeline.py --input file.laz` | Explicit input file |
| `python pipeline.py --input file.laz --scan-type ULS` | Use ULS preset (drone/aerial scans) |

`--scan-type` options: `TLS` (default, terrestrial ground scan) or `ULS` (drone/aerial scan).

### single_tree.py

| Command | Description |
|---|---|
| `python single_tree.py --build` | Compile AdTree — run this once before first use |
| `python single_tree.py` | Auto-detect input file from `input/` folder |
| `python single_tree.py --input file.laz` | Explicit input file |
| `python single_tree.py --input file.laz --no-filter` | Skip noise filtering |

Supported input formats: `.las`, `.laz`, `.ply`, `.txt`, `.csv`

---

## Output Structure

### Forest pipeline
```
output/
    {cloud_id}_results.zip
    pointtree/
        {cloud_id}_labeled.laz         full cloud with instance_id per point
        {cloud_id}_assembly.json        tree metadata and scene origin
        {cloud_id}_tree_summary.csv     per-tree stats (height, trunk diam, n_points)
        individual_trees/
            tree_0001.laz
            tree_0001_meta.json
            ...
    adtree/
        tree_0001/
            tree_0001_pointcloud.ply
            tree_0001_skeleton.ply
            tree_0001_branches.obj
            tree_0001_leaves.obj
            tree_0001_leaves_filtered06.obj   60% leaf density
            tree_0001_leaves_filtered03.obj   30% leaf density
        forest/
            forest_pointcloud.ply
            forest_branches.obj
            forest_leaves.obj
            forest_leaves_filtered06.obj
            forest_leaves_filtered03.obj
```

### Single tree
```
output/
    {tree_id}/
        {tree_id}_pointcloud.ply
        {tree_id}_skeleton.ply
        {tree_id}_branches.obj
        {tree_id}_leaves.obj
        {tree_id}_leaves_filtered06.obj
        {tree_id}_leaves_filtered03.obj
    {tree_id}_results.zip
```

---

## License

This project is released for research and educational use.

AdTree is licensed under **GPL v3**. Since this project builds on AdTree, the same license applies here. This means: you may freely use, modify and share this code, but any published derivative work must also be GPL v3 and include the source code.

PointTree retains its own license — see [PointTree license](https://github.com/ai4trees/pointtree/blob/main/LICENSE).

**TreeNet3D data is not included in this repository.** The dataset requires a signed agreement with the authors and may not be redistributed. See [TreeNet3D](https://github.com/ao216/TreeNet3D) for access instructions.
