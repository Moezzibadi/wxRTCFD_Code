# Real-time Computational Fluid Dynamics (ML-Enhanced - GPU Version)

Machine Learning Enhanced Real-Time CFD — currently under dev and training by Mohammad Moezzibadi. A ML Module is integrated but the training part is not public yet.

**This repository is a fork of [skhelladi/wxRTCFD_Code](https://github.com/skhelladi/wxRTCFD_Code), originally developed by Sofiane KHELLADI.** 
The original OpenMP parallelized core and the wxWidgets graphical interface were authored by Sofiane KHELLADI.

The project extends the real-time CFD solver with machine learning correction for adaptive flow prediction and enhanced real-time performance.

![ML Correction Results](final_comparison_results.png)

*Note: The ML correction is trained on a single thread (no parallelization) to ensure bit-perfection during the training process. This GPU version is specifically adapted for **macOS** using **Metal shaders** for parallel computation (as a proof of concept). It has been tested and verified on macOS hardware (Intel Iris Plus Graphics 645 1536 MB).*

Before building the project, make sure the following libraries are installed:
- **wxWidgets**
- **Metal (macOS Native)**
- **LibTorch**

## Features
This project is a real-time CFD solver based on a "rough" representation of conservation equations. The solver is implemented in **C/C++** (Objective-C++ for Metal) for **real-time** purpose and **wxWidgets** for the user interface and graphical renderings.
- 2D problems (3D in progress)
- Real-time flow pattern variation
- Variety of obstacles
- Postprocessing using: scalars (pressure, velocity, tracer), streamlines and velocity vectors
- **GPU Acceleration**: Core physics (Incompressibility, Integration, Extrapolation) implemented via Metal Compute Shaders.

## Install and build (macOS)
### Using cmake (command-line)
```bash
mkdir build
cd build
# set CMAKE_PREFIX_PATH to your LibTorch location
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j
```

## Authors
- **Mohammad Moezzibadi** (ML Correction & Metal GPU implementation)
- **Sofiane KHELLADI** (Original OpenMP Solver & wxWidgets UI)

## License
This project is licensed under the GPL-3 license.

### Code inspiration
This code is based on the theoretical developments and javascript code presented by Matthias Müller in "Ten Minute Physics" channel.
Link: https://matthias-research.github.io/pages/tenMinutePhysics/17-fluidSim.pdf
