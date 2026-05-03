# Real-time Computational Fluid Dynamics (ML-Enhanced)

Machine Learning Enhanced Real-Time CFD — currently under dev and training by Mohammad Moezzibadi. A ML Module is integrated but the training part is not public yet.

**This repository is a fork of [skhelladi/wxRTCFD_Code](https://github.com/skhelladi/wxRTCFD_Code), originally developed by Sofiane KHELLADI.** 
The original OpenMP parallelized core and the wxWidgets graphical interface were authored by Sofiane KHELLADI.

The project extends the real-time CFD solver with machine learning correction for adaptive flow prediction and enhanced real-time performance.

### ML Correction Performance Demo
![ML Correction 5000 Cells](MLCorrection_5000Cells.gif)
*Showcasing real-time ML correction for flow prediction.*

![ML Correction Results](final_comparison_results.png)

### Metal GPU Performance Demo (GPU Version)
![Metal GPU 1 Million Cells](pics/GPU_1millionCells.gif)
*Showcasing real-time simulation at high resolution (1,000,000 cells) using Metal Compute Shaders.*

*Note: The ML correction is trained on a single thread (no parallelization) to ensure bit-perfection during the training process, but it is implemented in the GPU version (see `GPU_version` branch) as a proof of concept. It has been tested and verified on macOS hardware (GPU 1.5 GB).*

Before building the project, make sure the following libraries are installed:
- **wxWidgets**
- **OpenMP**
- **LibTorch**

## Solver Determinism & Bit-Perfection
To support **Solver-in-the-Loop (SITL)** training, the core C++ solver and the ML correction module are currently configured for **single-threaded (serial) execution**. 

While the original solver by Sofiane KHELLADI is fully parallelized with OpenMP, these pragmas have been commented out in the current branch for the following reasons:
1. **Mathematical Bit-Perfection**: Parallelizing floating-point operations (like Gauss-Seidel iterations or Advection) introduces non-deterministic rounding errors due to the non-associativity of floating-point addition. This "numerical noise" ($10^{-16}$) can destabilize the Adjoint Method used during training.
2. **Gradient Consistency**: SITL requires a bit-perfect match between the C++ forward pass and the Python-based differentiable solver (`FluidTorch_Batch_BitPerfect.py`).
3. **ML Inference**: The ML correction step is also serial to maintain strict alignment with the training environment.

For high-performance production runs where training is not required, OpenMP can be re-enabled, or the **Metal GPU version** (found in the `GPU_version` branch) can be used for massive parallelization.

## Features
This project is a real-time CFD solver based on a "rough" representation of conservation equations. The solver is implemented in **C/C++** for **real-time** purpose and **wxWidgets** for the user interface and graphical renderings. It supports:
- 2D problems (3D in progress)
- Real-time flow pattern variation
- Variety of obstacles
- Postprocessing using: scalars (pressure, velocity, tracer), streamlines and velocity vectors

## Install and build
### Using cmake (command-line)
```bash
mkdir build
cd build
# set CMAKE_PREFIX_PATH to your LibTorch location
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/path/to/libtorch
make -j
```

## Authors
- **Mohammad Moezzibadi** (ML Correction & GPU implementation)
- **Sofiane KHELLADI** (Original OpenMP Solver & wxWidgets UI)

## License
This project is licensed under the GPL-3 license.

### Code inspiration
This code is based on the theoretical developments and javascript code presented by Matthias Müller in "Ten Minute Physics" channel.
Link: https://matthias-research.github.io/pages/tenMinutePhysics/17-fluidSim.pdf
