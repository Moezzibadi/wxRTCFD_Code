
# Real-time Computational Fluid Dynamics (ML-Enhanced)

**Machine Learning Enhanced Real-Time CFD** — currently under dev and training by **Mohammad Moezzibadi**. A ML Module is integrated but the training part is not public yet.

This repository is a fork of [skhelladi/wxRTCFD_Code](https://github.com/skhelladi/wxRTCFD_Code),  
originally developed by **Sofiane KHELLADI**.

The project extends the real-time CFD solver with **machine learning** correction for  
adaptive flow prediction and enhanced real-time performance.

Before building the project, make sure the following libraries are installed:

- wxWidgets
- OpenMP
- LibTorch

---

## Original Description

# Real-time Computational Fluid Dynamics

This repository presents a real-time CFD solver based on a "rough" representation of conservation equations. The solver is implemented in **C/C++** for **real-time** purpose and **wxWidget** for the user interface and graphical renderings. It supports a wide range of features:
- 2D problems (3D in progress)
- real-time flow patern variation
- variety of obstacles (in progress)
- drag-and-drop obstacles
- postprocessing using: scalars (pressure, velocity, tracer), streamlines and velocity vectors

## Getting Started
 	
### Prerequisites

First, make sure the following libraries are installed. Mainly

- wxWidgets
- OpenMP

### Install an build
#### 1. Using cmake (command-line)
```
git clone https://github.com/skhelladi/wxRTCFD_Code.git
cd wxRTCFD_Code
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
```

#### 2. Using CodeBlocks
```
git clone https://github.com/skhelladi/wxRTCFD_Code.git
cd wxRTCFD_Code 
codeblocks wxRTCFD_Code.cbp
```
then build.

#### 3. Using Visual Studio Code 
```
git clone https://github.com/skhelladi/wxRTCFD_Code.git
cd wxRTCFD_Code
code .
```
then build.

### Run the code
Execute wxRTCFD_Code binary file in build directory.


## Screenshots
________________________
<img src="doc/fig_1.png" width="400" height="200" />    <img src="doc/fig_2.png" width="400" height="200" /> 
<img src="doc/fig_3.png" width="400" height="200" /> <img src="doc/fig_4.png" width="400" height="200" /> 
<img src="doc/fig_6.png" width="400" height="200" /> <img src="doc/fig_7.png" width="400" height="200" />
_______________________

## Tutorial (on Youtube)

[![Tutorial](doc/fig_4.png)](http://www.youtube.com/watch?feature=player_embedded&v=hqhZNt9UP4Q)

## License
This project is licensed under the GPL-3 license.

Unless you explicitly state otherwise, any contribution intentionally submitted by you for inclusion in this project shall be licensed as above, without any additional terms or conditions.

## Authors
- Sofiane KHELLADI


### Code inspiration
This code is based on the theoretical developments and javascript code presented by Matthias Müller in "Ten Minute Physics" channel.

Link: https://matthias-research.github.io/pages/tenMinutePhysics/17-fluidSim.pdf

and the Qt version of the same code developed by Sofiane KHELLADI

Link: https://github.com/skhelladi/RTCFD_Code
