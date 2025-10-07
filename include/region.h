#ifndef REGION_H
#define REGION_H

#include <iostream>
#include <memory>
#include <functional>
#include "fluid.h"

#ifdef USE_LIBTORCH
#include <torch/script.h>
#endif

#define DEBUG 1

enum OBJ
{
    CYLINDER,
    SQUARE,
    DIAMOND,
    NACA,
    ROTOR
};

using namespace std;

class Region
{
public:
    // Constructor
    Region(int _height, int _width, double _simHeight = 1.1);

    // Coordinate conversion
    double cX(double x) { return x * cScale; }
    double cY(double y) { return height - y * cScale; }

    // Region setup
    void setupRegion(int _RegionNr = 0, double _overRelaxation = 1.9,
                     int _resolution = 50, double _density = 1000,
                     int _numThreads = 1);

    // Obstacle handling
    void setObstacle(double x, double y, bool reset);
    void setObstacleCylinder(double x, double y, bool reset);
    void setObstacleSquare(double x, double y, bool reset);
    void setObstacleDiamond(double x, double y, bool reset);
    void setObstacleNaca(double x, double y, bool reset);
    void setObstacleRotor(double x, double y, bool reset);

    // Region size update
    void updateRegionSize(int _height, int _width);

    // Simulation update
    void update();

    // Function for ML correction step
    std::function<void(Fluid&)> correctionStep;

    // ---------------- Simulation parameters -----------------
    string text = "";
    double gravity = -9.81;
    double dt = 1.0 / 60.0;
    int numIters = 40;
    int frameNr = 0;
    double overRelaxation = 1.9;
    double obstacleX = 0.0;
    double obstacleY = 0.0;
    double characteristic_length = 0.15;
    bool paused = false;
    int RegionNr = 0;

    // ---------------- ML correction -----------------
    bool mlCorrectionEnabled = false;
    #ifdef USE_LIBTORCH
        torch::jit::script::Module mlModel; // neural network module

        void setMLCorrection(bool enabled, const std::string &modelPath)
        {
            mlCorrectionEnabled = enabled;
            if (enabled)
            {
                try
                {
                    mlModel = torch::jit::load(modelPath);
                    mlModel.eval();
                    std::cout << "ML model loaded from: " << modelPath << std::endl;
                    correctionStep = [this](Fluid &f) { f.applyCorrection(mlModel, 3.0); };
                }
                catch (const c10::Error &e)
                {
                    std::cerr << "Error loading the model: " << e.what() << std::endl;
                    correctionStep = [this](Fluid &f) { f.NoCorrection(); };
                }
            }
            else
            {
                std::cerr << "No ML Correction! " << std::endl;
                correctionStep = [this](Fluid &f) { f.NoCorrection(); };

            }
        }
    #endif

    // ---------------- GUI flags -----------------
    bool showObstacle = false;
    bool showObstaclePosition = false;
    bool showStreamlines = false;
    bool showVelocity = false;
    bool showXVelocity = false;
    bool showYVelocity = false;
    bool showVelocityVectors = false;
    bool showPressure = false;
    bool showTracer = true;
    bool useNetworkCorrection = false;

    // ---------------- Simulation data -----------------
    shared_ptr<Fluid> fluid;

    // Region geometry
    int height;
    int width;
    double simHeight;
    double cScale;
    double simWidth;
    int resolution;
    int numThreads;

    OBJ obstacle;
};

// Utility function
OBJ indexToOBJ(int index);

#endif // REGION_H
