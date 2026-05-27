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

### Forest Pipeline (`pipeline.py` / `Forest_Reconstruction_Pipeline.ipynb`)

Takes a segmented forest point cloud as input and runs a two-stage pipeline:

1. **PointTree** segments the full forest scan into individual tree instances, detects trunk positions and estimates trunk diameters.
2. **AdTree** reconstructs each individual tree — generating a skeleton, branch mesh and leaf mesh.

The outputs are merged back into a complete forest and packaged as a ZIP.


<img width="122" height="402" alt="im1" src="https://github.com/user-attachments/assets/1ae16a3b-62ee-43a6-a975-6fe8833666ce" />


### Single Tree Pipeline (`single_tree.py` / `single_tree_reconstruction.ipynb`)

Takes a single-tree point cloud as input and runs AdTree directly. Designed for testing and validation of individual trees. Accepts any `.las`, `.laz`, `.ply`, `.txt` or `.csv` file.

---

## Changes Made to AdTree

The original AdTree C++ source was modified with four patches applied before compilation. All patches are applied automatically by the pipeline scripts.

### Patch A — Leaf density and size reduction

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

### Patch B — Elliptic leaf shape

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

### Patch C — Leaf base attached to branch tip

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

### Patch D — Improved initial trunk radius estimate

**Context from the paper:** AdTree uses a Levenberg-Marquardt non-linear least-squares cylinder fit (Section 3.3, Equations 5–7) to accurately determine the trunk radius. This 3D cylinder fit is the core of the original algorithm and is left unchanged. However, this fit requires a good initial estimate to converge correctly. In the original code, this initial estimate comes from the 2D bounding box of trunk points.

**Before:** The initial trunk radius estimate used the 2D bounding box of trunk points projected onto the XY plane — sensitive to outliers and elongated cross-sections.
```cpp
TrunkRadius_ = std::max((maxX - minX), (maxY - minY)) / 2.0;
```

**After:** A 2D Gauss-Newton least-squares circle fit replaces the bounding box as the initial estimate. It runs for up to 100 iterations using Cramer's rule to solve the 3x3 normal equations, and converges to a better starting value for the subsequent 3D cylinder fit. Applied only when ≥ 1000 trunk points are available.

**Why:** The bounding box overestimates the radius when trunk cross-sections are slightly elongated or contain outliers. A better starting estimate helps the 3D Levenberg-Marquardt cylinder fit (which remains unchanged) converge to a more accurate result, which then propagates to all branch radii via the allometric scaling rule (Equation 8 in the paper).

---

### Additional: trunk list threshold (epsiony)

**Before:** Only points within 2% of tree height from the lowest point were used for trunk analysis.

**After:** The threshold is raised to 10%, providing more points for robust trunk radius estimation.

```cpp
// before
double epsiony = 0.02;
// after
double epsiony = 0.10;
```

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
    Forest_Reconstruction_Pipeline.ipynb
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

### Option 4 — Linux (Ubuntu / Debian)

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
