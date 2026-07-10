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



| :---: | :---: | :---: | :---: |
| <img src="https://github.com/user-attachments/assets/7ae4b63a-79d4-4911-bf63-38eb9e514418" alt="Point Cloud" width="180"> | <img src="https://github.com/user-attachments/assets/224744a4-bc23-4c1a-9ae1-b65299907390" alt="Tree instance segmentation" width="180"> | <img src="https://github.com/user-attachments/assets/9c7799b4-d01d-442f-9091-ba6f644e164b" alt="AdTree wood mesh" width="180"> | <img src="https://github.com/user-attachments/assets/87c7a51c-75fd-4d2d-b651-209fbd101188" alt="AdTree leaf mesh" width="180"> |
| Point Cloud | Tree instance segmentation | AdTree wood mesh | After |

### Single Tree Pipeline (`single_tree.py` / `single_tree_reconstruction.ipynb`)

Takes a single-tree point cloud as input and runs AdTree directly. Designed for testing and validation of individual trees. Accepts any `.las`, `.laz`, `.ply`, `.txt` or `.csv` file.

---

## Changes Made to AdTree

The original AdTree C++ source is modified with nine patches applied automatically before compilation. The patches are grouped and ordered to mirror the three reconstruction stages described in the paper — **skeleton extraction**, **wood-mesh reconstruction** and **leaf generation** — and are numbered consecutively (1–9) across the groups. All patches are applied automatically by the pipeline scripts.

> **Images:** Each patch starts with a before/after comparison. Replace the placeholder paths (`docs/images/patchN_before.png` / `docs/images/patchN_after.png`) with your own screenshots.

### 1. Skeleton Extraction

#### Patch 1 — Location-dependent skeleton simplification

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 1 – before](docs/images/patch1_before.png) | ![Patch 1 – after](docs/images/patch1_after.png) |

**Before:** A single fixed merge threshold was used for the whole tree when simplifying the skeleton — a vertex was merged whenever the deviation was below `1.0 * r`. This simplified the trunk region too aggressively and oversimplified the main structure.
```cpp
double r = (*i_Graph)[edge(i_dVertex, parentV, *i_Graph).first].nRadius;
if (distance >= 1.0 * r)
    return false;
```

**After:** The merge threshold becomes location-dependent, driven by the node's position in the tree (`fraction = lengthOfSubtree(node) / lengthOfSubtree(root)`, ~1 at the trunk, ~0 at the crown tips). A smoothstep over the band `[0.05, 0.25]` relaxes merging at the trunk (threshold `≈ 0.1`, detail preserved) and keeps it aggressive toward the crown (threshold `1.0`).
```cpp
double fraction = (rootSubtree > 1e-10) ? (nodeSubtree / rootSubtree) : 0.0;
double bandLo = 0.05, bandHi = 0.25, curveStrength = 0.1;
double tt = clamp((fraction - bandLo) / (bandHi - bandLo), 0.0, 1.0);
double s  = tt*tt*(3.0 - 2.0*tt);                 // smoothstep
double mergeThreshold = 1.0 + (curveStrength - 1.0) * s;
if (distance >= mergeThreshold * r)
    return false;
```

**Why:** The uniform simplification was too aggressive in the trunk region, oversimplifying the main stem (paper Fig. 2b). Making the decision rule position-dependent preserves trunk and main-branch detail while still simplifying the crown, as described in Section III-A of the paper.

---

#### Patch 2 — Junction-aware skeleton smoothing & gap filling

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 2 – before](docs/images/patch2_before.png) | ![Patch 2 – after](docs/images/patch2_after.png) |

**Before:** The reconstructed centerline and radii were taken directly from the cubic interpolation of each branch path. This could produce wavy centerlines, jittery radius profiles and long straight jumps where the interpolated points were sparse.

**After:** Each branch path is tagged with `hardAnchors` (root, junctions and branch tips that must not move), then post-processed:
- **Centerline smoothing** — 4 passes of windowed averaging that never crosses a hard anchor.
- **Radius smoothing** — 8 passes plus a monotonic non-increasing constraint toward the tip.
- **Straight-gap filling** — inserts evenly spaced points wherever the spacing between consecutive skeleton points exceeds `1.4 ×` the median spacing.

Junctions and tips stay fixed throughout, so the topology is preserved.

**Why:** Smoother centerlines and radius profiles give more natural, less faceted branch cylinders, and gap filling avoids long straight cylinder segments where the interpolation was sparse — without moving the structural anchor points.

---

#### Patch 3 — Lower trunk straightening

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 3 – before](docs/images/patch3_before.png) | ![Patch 3 – after](docs/images/patch3_after.png) |

**Before:** The lowest section of the main trunk followed the raw skeleton, which could wobble or lean near the base.

**After:** On the main path only, the lowest ~5% of the trunk (by height) is straightened. An attachment point is found at `5% × TreeHeight`, a least-squares line direction is estimated from the next few points above it, and the points below are projected onto that line while keeping their original heights (no offset introduced).

**Why:** The base of the stem is where reconstruction artifacts are most visible. Aligning the lowest 5% to the local trunk direction estimated just above the base gives a cleaner, more vertical stem.

---

### 2. Wood-Mesh Reconstruction

#### Patch 4 — Trunk-point threshold raised from 2% to 10% (epsiony)

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 4 – before](docs/images/patch4_before.png) | ![Patch 4 – after](docs/images/patch4_after.png) |

**Before:** Only points within 2% of tree height from the lowest point were used for trunk analysis.

**After:** The threshold is raised to 10%, providing more points for robust trunk radius estimation.

```cpp
// before
double epsiony = 0.02;
// after
double epsiony = 0.10;
```

---

#### Patch 5 — Improved initial trunk radius estimate (least-squares circle fit)

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 5 – before](docs/images/patch5_before.png) | ![Patch 5 – after](docs/images/patch5_after.png) |

**Context from the paper:** AdTree uses a Levenberg-Marquardt non-linear least-squares cylinder fit (Section 3.3, Equations 5–7) to accurately determine the trunk radius. This 3D cylinder fit is the core of the original algorithm and is left unchanged. However, this fit requires a good initial estimate to converge correctly. In the original code, this initial estimate comes from the 2D bounding box of trunk points.

**Before:** The initial trunk radius estimate used the 2D bounding box of trunk points projected onto the XY plane — sensitive to outliers and elongated cross-sections.
```cpp
TrunkRadius_ = std::max((maxX - minX), (maxY - minY)) / 2.0;
```

**After:** A 2D Gauss-Newton least-squares circle fit replaces the bounding box as the initial estimate. It runs for up to 100 iterations using Cramer's rule to solve the 3x3 normal equations, and converges to a better starting value for the subsequent 3D cylinder fit. Applied only when ≥ 1000 trunk points are available.

**Why:** The bounding box overestimates the radius when trunk cross-sections are slightly elongated or contain outliers. A better starting estimate helps the 3D Levenberg-Marquardt cylinder fit (which remains unchanged) converge to a more accurate result, which then propagates to all branch radii via the allometric scaling rule (Equation 8 in the paper).

---

#### Patch 6 — Self-calibrating final trunk radius

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 6 – before](docs/images/patch6_before.png) | ![Patch 6 – after](docs/images/patch6_after.png) |

**Before:** The trunk radius produced by the earlier estimate/fit was passed directly to `compute_all_edges_radius(TrunkRadius_)`, which propagates it to every branch.

**After:** Just before that propagation, `TrunkRadius_` is recomputed directly from the point cloud: skeleton points within the lowest 2% of tree height are collected, their XY centroid is taken, and `TrunkRadius_` is set to the **median** radial distance of those points from the centroid (only when ≥ 10 points are available).

**Why:** This anchors the final trunk radius to the actual point distribution at the base, reducing over/under-estimation right before the radius is scaled up the rest of the tree. *(Note: this median override runs after Patch 5 — see the note in the pull-request/README feedback about their interplay.)*

---

### 3. Leaf Generation

#### Patch 7 — Leaf density and size reduction

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 7 – before](docs/images/patch7_before.png) | ![Patch 7 – after](docs/images/patch7_after.png) |

**Before:** Each end vertex generated up to 10 leaves with a large leaf radius and size.
```cpp
int density = ceil(random_float() * 10);
generate_leaves(currentLeafVertex, 0.05);
double radius = 0.2 / log((float)num_edges(simplified_skeleton_));
```

**After:** Density reduced to at most 1 leaf per end vertex, size and radius scaled down significantly.
```cpp
int density = ceil(random_float() * 1);
generate_leaves(currentLeafVertex, 0.02);
double radius = 0.04 / log((float)num_edges(simplified_skeleton_));
```

**Why:** The original settings produced extremely dense, oversized leaf meshes that were visually unrealistic and very large in file size.

---

#### Patch 8 — Leaf base attached to branch tip

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 8 – before](docs/images/patch8_before.png) | ![Patch 8 – after](docs/images/patch8_after.png) |

**Before:** Leaf position was randomly placed along the direction from the branch tip toward its parent, effectively scattering leaves away from the actual branch endpoint.
```cpp
vec3 pEnd = pCurrent - (random_float() / 2.0) * ((pCurrent - pParent).normalize());
vec3 dirLeaf = random_direction();
vec3 pLeaf = pEnd + dirLeaf * random_float() * radius;
```

**After:** The leaf base is placed directly at the branch tip with a small offset along the branch direction. The leaf grows outward perpendicular to the branch, blended slightly with the branch direction for a natural draping effect.
```cpp
vec3 branchDir = (pCurrent - pParent).normalize();
double offset = random_float() * radius * 0.5;
vec3 pLeaf = pCurrent - branchDir * offset;
vec3 dirLeaf = (randPerp * 0.6f + branchDir * 0.4f).normalize();
```

**Why:** Leaves were floating away from branches instead of growing from them. This fix anchors the leaf base to the branch endpoint where it belongs.

---

#### Patch 9 — Elliptic leaf shape

<!-- Before/After image -->
| Before | After |
| :---: | :---: |
| ![Patch 9 – before](docs/images/patch9_before.png) | ![Patch 9 – after](docs/images/patch9_after.png) |

**Before:** Each leaf was a flat quad (two triangles), producing rectangular leaves with no shape variation.

**After:** Each leaf is constructed as an elliptic strip with 6 segments and a sine-profile width function, producing a natural tapered leaf shape.

```cpp
const int nSegs = 6;
for (int s = 0; s <= nSegs; ++s) {
    double t = (double)s / nSegs;
    double width = sin(M_PI * t) * (2.0 - 0.3 * t);  // tapered elliptic profile
    ...
}
```

**Why:** Flat rectangular quads look unnatural. The elliptic profile gives leaves a realistic pointed tip and wider mid-section.

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
