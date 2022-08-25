# SUbStack ANalysis (SUSAN): High performance Subtomogram Averaging
**(Personal/Development version)**

`SUSAN` is a low-level/mid-level framework for fast Subtomogram Averaging (StA) for CryoEM. It uses *susbtacks* instead of *subtomograms* that are cropped *on-the-fly* from the aligned stacks to reduce the computational complexity and to increase the overall performace of the StA pipeline.

`SUSAN` was designed to be modular, flexible and fast. It is conformed by two layers:
- **Low-level layer**: Set of executables that perform the demanding computations. They were written in `C++` using a minimal set of dependencies: `Eigen`, as a header-only mathemetical engine, `CUDA` for GPU acceleration, and `PThreads` for lightweight multi-threading. Optionally, they can be built with [`MPI`](https://en.wikipedia.org/wiki/Message_Passing_Interface) support to run in multi-node environments.
- **Mid-level layer**: Set of wrappers to the previous layer that simplify its use and provides a set of non time-critical operations. It is used to create the workflows or pipelines as scripts with wrappers for `Matlab` and for `Python`. The `Matlab` one was designed to complement [DYNAMO](https://wiki.dynamo.biozentrum.unibas.ch/w/index.php/Main_Page), while the `Python` one is provided to enable integration to other pipelines based on this language.

### About
I started the development of `SUSAN` at the [Independent Research Group (Sofja Kovaleskaja) of Dr. Misha Kudryashev](https://www.biophys.mpg.de/2149775/members) at the Department of Structural Biology [Max Planck Institute of Biophysics (MPIBP)](https://www.biophys.mpg.de/en) in Frankfurt am Main, Germany. Currently, I am an ARISE fellow at the [Kreshuk group](https://www.embl.org/groups/kreshuk/members/) at the [European Molecular Biology Laboratory (EMBL)](https://www.embl.org/) in Heidelberg, Germany. Dr. Misha Kudryashev has a new [group](https://www.mdc-berlin.de/kudryashev) at the [Max Delbrück Center of Molecular Medicine (MDCMM)](https://www.mdc-berlin.de/) in Berlin, Germany.

`SUSAN` is an Open Source project ([AGPLv3.0](LICENSE))

## Building and setup instructions
### Dependencies
- `CUDA`.
- `Eigen`: As this is needed at compilation time only, there is no need to install it in the system. If not provided by the building environment, it can be installed locally following [this](extern/README.md) instructions.
- `gcc`.
- `cmake`.
- `git`.
#### Optional
- `OpenMPI`, or equivalent, for multi-node support
- `Matlab`

### Initial setup and compilation
We assume that `SUSAN` will be installed in the `LOCAL_SUSAN_PATH` folder (`LOCAL_SUSAN_PATH` can be `/home/user/Software/`, for example)
1. Install the desired dependencies.
2. Clone `SUSAN` to `LOCAL_SUSAN_PATH`:
```
cd LOCAL_SUSAN_PATH
git clone https://github.com/rkms86/SUSAN
```
   - **(Optional)** Install `Eigen`:
```
cd LOCAL_SUSAN_PATH/extern
git clone https://gitlab.com/libeigen/eigen.git eigen
cd eigen
mkdir build
cd build
cmake ../ -DCMAKE_INSTALL_PREFIX=../../eigen_lib
make install
cd LOCAL_SUSAN_PATH
```
3. Compile `SUSAN`:
```
mkdir bin
cd bin
cmake ../
make -j
```
**Note:** The `cmake` procedure detects the availabilty of `OpenMPI` and `Matlab` and compiles their functionalities accordingly.

- **(Optional)** To be able to install `SUSAN` as a `Python` package:
```
make prepare_python
```
This will install the compiled binaries in the `bin` folder of the `Python` package.

### `Python` setup
#### Dependencies
Besides the standard libraries, the `SUSAN` module for `Python` has only two dependencies: [`NumPy`](https://numpy.org/) and [`Numba`](https://numba.pydata.org/). Install them if needed:
- Using [`conda`](https://conda.io) (or equivalent):
```
conda install numpy numba
```
- Using [`pip`](https://pypi.org/)
```
pip install numpy numba
```

#### Option 1: Using `susan` without installation
`LOCAL_SUSAN_PATH` must be added to path first and then the `susan` module can be imported. On the `Python` command line, or on a `Python` script:
```
import sys
sys.path.insert(1,'LOCAL_SUSAN_PATH')
import susan
```

#### Option 2: Install `susan` in the current `Python` environment
After executing the optional steps for the `Python` module, it can be installed using [`pip`](https://pypi.org/) and the provided [setup.py](setup.py) file. In the  `LOCAL_SUSAN_PATH` execute:
```
pip install .
```
After this step, the module `susan` should be available to be imported.

### `Matlab` setup
`LOCAL_SUSAN_PATH` must be added to path. On the `Matlab` command line, or on a `Matlab` script:
```
addpath LOCAL_SUSAN_PATH
```

## Tutorial
A tutorial is available for `Python` and `Matlab` for the `mixedCTEM` dataset from the [EMPIAR-10064](https://www.ebi.ac.uk/empiar/EMPIAR-10064/). It is assumed that the `wget` and `gunzip` commands and the `IMOD` framework are installed in the system.

### Preparing the data
1. Download the dataset (uses `wget`):
```
cd LOCAL_SUSAN_PATH/tutorials/empiar_10064/data
./download_data.sh
```
2. Create the aligned stacks (uses `IMOD`):
```
./create_binned_aligned_stacks.sh
```
3. Uncompress the initial reference (uses `gunzip`):
```
cd LOCAL_SUSAN_PATH/tutorials/empiar_10064
gunzip emd_3420_b4.mrc.gz
```

### Running the Tutorial
Depending on the system setup:
- For `Matlab` use [workflow.m](tutorials/empiar_10064/workflow.m).
- For `Python` use [workflow.ipynb](tutorials/empiar_10064/workflow.ipynb) (as a [Jupyter Notebook](https://jupyter.org/install)).






