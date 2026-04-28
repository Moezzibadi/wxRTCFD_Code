#include "region.h"
#include "settings.h"
#include <sstream>
#include <fstream>


Region::Region(int _height, int _width, double _simHeight)
{
    height = _height;
    width = _width;
    simHeight = _simHeight;
    cScale = height / simHeight;
    simWidth = width / cScale;
    obstacle = CYLINDER;

// #if (DEBUG)
//     cout << "------------------" << endl;
//     cout << "Region constructor" << endl;
//     cout << "------------------" << endl;
//     cout << "height = " << height << endl;
//     cout << "width = " << width << endl;
//     cout << "simHeight = " << simHeight << endl;
//     cout << "cScale = " << cScale << endl;
//     cout << "simWidth = " << simWidth << endl;
// #endif
}

void Region::setupRegion(int _RegionNr, double _overRelaxation, int _resolution, double _density, int _numThreads)
{
    this->RegionNr = _RegionNr;
    this->characteristic_length = 0.15;
    this->overRelaxation = _overRelaxation;
    this->numThreads = _numThreads;

    this->dt = 1 / 60.0;
    this->numIters = 40;

    this->obstacleX = 0;
    this->obstacleY = 0;
    this->resolution = _resolution;

    int res = this->resolution;

    if (RegionNr == 0)
        res = this->resolution;
    else if (RegionNr == 3)
        res = 2 * this->resolution;

    double domainHeight = 1.0;
    double domainWidth = domainHeight / simHeight * simWidth;
    double h = domainHeight / res;

    int numX = floor(domainWidth / h);
    int numY = floor(domainHeight / h);

#ifdef USE_LIBTORCH
    if (_resolution == 50) {
        numX = 88;
        numY = 52;
        if (mlCorrectionEnabled) {
            std::cout << "[REGION] ML Correction ENABLED" << std::endl;
            std::cout << "[REGION] Forcing grid to 88x52 for ML Bit-Perfection" << std::endl;
        } else {
            std::cout << "[REGION] ML Grid forced to 88x52 (Ready for ML)" << std::endl;
        }
    } else {
        if (mlCorrectionEnabled) {
            std::cout << "[REGION] ML Correction ENABLED" << std::endl;
            std::cout << "[REGION] Warning: Resolution is " << _resolution << ", but ML was trained at 50. Using interpolation." << std::endl;
        } else {
            std::cout << "[REGION] ML Correction DISABLED" << std::endl;
        }
    }
#endif

    std::cout << "[REGION] Requested Resolution: " << _resolution << " (RegionNr: " << _RegionNr << ")" << std::endl;
    std::cout << "[REGION] Calculated Solver Grid: numX = " << numX << ", numY = " << numY << " (h = " << h << ")" << std::endl;

    double density = _density;

    this->fluid = make_shared<Fluid>(density, numX, numY, h, overRelaxation, numThreads);
    shared_ptr<Fluid> f = this->fluid;

    #ifdef USE_LIBTORCH
        if (mlCorrectionEnabled) { // toggle from GUI
            // mlModel should already be loaded via setMLCorrection
            correctionStep = [this](Fluid &f) {
                f.applyCorrection(this->mlModel, this->inVel);
            };
        } else {
            correctionStep = [this](Fluid &f) { f.NoCorrection(); };
        }
    #else
        correctionStep = [this](Fluid &f) { f.NoCorrection(); };
    #endif

    // Sync corrected buffers initially
    memcpy(f->u_corrected.begin(), f->u.begin(), f->numCells * sizeof(float));
    memcpy(f->v_corrected.begin(), f->v.begin(), f->numCells * sizeof(float));

    int n = f->numY;
    RTCFDPoint pos;
    if (obstacle != ROTOR)
        pos = { 0.4, 0.5 };

    // --------------------- Region-specific initialization ---------------------
    if (RegionNr == 0) { // tank
#pragma omp parallel for schedule(static) num_threads(numThreads)
        for (int i = 0; i < f->numX; i++)
            for (int j = 0; j < f->numY; j++)
                f->s[i * n + j] = (i == 0 || i == f->numX - 1 || j == 0) ? 0.0 : 1.0;

        setObstacle(pos.x, pos.y, true);
        this->gravity = 0.0;
        this->showPressure = true;
        this->showTracer = false;
        this->showStreamlines = false;
        this->showVelocity = false;
        this->showXVelocity = false;
        this->showYVelocity = false;
        this->showVelocityVectors = false;
        this->showObstacle = true;
        this->showObstaclePosition = false;

    } else if (RegionNr == 1 || RegionNr == 3) { // vortex shedding
        double localInVel = this->inVel;
        std::cout << "[DEBUG] Setting up Region with inVel: " << localInVel << std::endl;
#pragma omp parallel for schedule(static) num_threads(numThreads)
        for (int i = 0; i < f->numX; i++)
            for (int j = 0; j < f->numY; j++) {
                f->s[i * n + j] = (i == 0 || j == 0 || j == f->numY - 1) ? 0.0 : 1.0;
                if (i <= 1)
                    f->u[i * n + j] = localInVel;
            }

        double pipeH = 0.1 * f->numY;
        int minJ = floor(0.5 * f->numY - 0.5 * pipeH);
        int maxJ = floor(0.5 * f->numY + 0.5 * pipeH);
        for (int j = minJ; j < maxJ; j++)
            f->m[j] = 0.0;

        setObstacle(pos.x, pos.y, true);

        this->gravity = 0.0;
        this->showPressure = false;
        this->showTracer = true;
        this->showStreamlines = false;
        this->showVelocity = false;
        this->showXVelocity = false;
        this->showYVelocity = false;
        this->showVelocityVectors = false;

        if (RegionNr == 3) {
            this->dt = 1 / 60.0;
            this->numIters = 40;
            this->showPressure = true;
        }
    } else if (RegionNr == 2) { // paint
        this->gravity = 0.0;
        this->overRelaxation = 1.0;
        this->showPressure = false;
        this->showTracer = true;
        this->showStreamlines = false;
        this->showVelocity = false;
        this->showXVelocity = false;
        this->showYVelocity = false;
        this->showVelocityVectors = false;
        this->characteristic_length = 0.075;
        this->showObstacle = true;
        this->showObstaclePosition = false;
        setObstacle(pos.x, pos.y, true);
    }
    
    if (this->fluid) {
        this->fluid->syncToTensors();
        memcpy(fluid->u_corrected.begin(), fluid->u.begin(), fluid->numCells * sizeof(float));
        memcpy(fluid->v_corrected.begin(), fluid->v.begin(), fluid->numCells * sizeof(float));
    }
}


void Region::setObstacle(double x, double y, bool reset)
{
    switch (obstacle)
    {
    case CYLINDER:
        setObstacleCylinder(x, y, reset);
        break;
    case SQUARE:
        setObstacleSquare(x, y, reset);
        break;
    case DIAMOND:
        setObstacleDiamond(x, y, reset);
        break;
    case NACA:
        setObstacleNaca(x, y, reset);
        break;
    case ROTOR:
        setObstacleRotor(x, y, reset);
        break;
    default:
        setObstacleCylinder(x, y, reset);
    }
}

void Region::setObstacleCylinder(double x, double y, bool reset)
{

    double vx = 0.0;
    double vy = 0.0;

    if (!reset)
    {
        vx = (x - this->obstacleX) / this->dt;
        vy = (y - this->obstacleY) / this->dt;
    }

    this->obstacleX = x;
    this->obstacleY = y;
    double r = this->characteristic_length;
    shared_ptr<Fluid> f = this->fluid;
    int n = f->numY;
    //    double cd = sqrt(2) * f->h;
#pragma omp parallel for schedule(static) num_threads(fluid->numThreads)
    for (int i = 1; i < f->numX - 2; i++)
    {
        for (int j = 1; j < f->numY - 2; j++)
        {

            f->s[i * n + j] = 1.0;

            double dx = (i + 0.5) * f->h - x;
            double dy = (j + 0.5) * f->h - y;

            if (dx * dx + dy * dy < r * r)
            {
                f->s[i * n + j] = 0.0;
                if (this->RegionNr == 2)
                    f->m[i * n + j] = 0.5 + 0.5 * sin(0.1 * this->frameNr);
                else
                    f->m[i * n + j] = 1.0;

                f->u[i * n + j] = vx;
                f->u[(i + 1) * n + j] = vx;
                f->v[i * n + j] = vy;
                f->v[i * n + j + 1] = vy;
            }
        }
    }

    this->showObstacle = true;
    if (f) f->syncToTensors();
}

void Region::setObstacleSquare(double x, double y, bool reset)
{

    double vx = 0.0;
    double vy = 0.0;

    if (!reset)
    {
        vx = (x - this->obstacleX) / this->dt;
        vy = (y - this->obstacleY) / this->dt;
    }

    this->obstacleX = x;
    this->obstacleY = y;
    double r = this->characteristic_length;
    shared_ptr<Fluid> f = this->fluid;

    vector<RTCFDPoint> P = getSquarePoints(RTCFDPoint({x, y}), r);

    int n = f->numY;
#pragma omp parallel for schedule(static) num_threads(f->numThreads)
    for (int i = 1; i < f->numX - 2; i++)
    {
        for (int j = 1; j < f->numY - 2; j++)
        {

            f->s[i * n + j] = 1.0;

            // double dx = (i + 0.5) * f->h - x;
            // double dy = (j + 0.5) * f->h - y;

            // if (fabs(dx)<r&&fabs(dy)<r)
            if (isInsidePolygon(P, RTCFDPoint({(i + 0.5) * f->h, (j + 0.5) * f->h})))
            {
                f->s[i * n + j] = 0.0;
                if (this->RegionNr == 2)
                    f->m[i * n + j] = 0.5 + 0.5 * sin(0.1 * this->frameNr);
                else
                    f->m[i * n + j] = 1.0;

                f->u[i * n + j] = vx;
                f->u[(i + 1) * n + j] = vx;
                f->v[i * n + j] = vy;
                f->v[i * n + j + 1] = vy;
            }
        }
    }

    this->showObstacle = true;
    if (f) f->syncToTensors();
}

void Region::setObstacleDiamond(double x, double y, bool reset)
{
    double vx = 0.0;
    double vy = 0.0;

    if (!reset)
    {
        vx = (x - this->obstacleX) / this->dt;
        vy = (y - this->obstacleY) / this->dt;
    }

    this->obstacleX = x;
    this->obstacleY = y;
    double r = this->characteristic_length;
    shared_ptr<Fluid> f = this->fluid;
    int n = f->numY;
    RTCFDPoint center = {x, y};
    vector<RTCFDPoint> P = getDiamondPoints(center, r);
    //    double cd = sqrt(2) * f->h;
#pragma omp parallel for schedule(static) num_threads(f->numThreads)
    for (int i = 1; i < f->numX - 2; i++)
    {
        for (int j = 1; j < f->numY - 2; j++)
        {

            f->s[i * n + j] = 1.0;

            // double dx = (i + 0.5) * f->h - x;
            // double dy = (j + 0.5) * f->h - y;

            // //! axis change by a rotation of theta=pi/4
            // double dxb = sqrt(2) / 2 * (dx + dy);
            // double dyb = sqrt(2) / 2 * (-dx + dy);

            // if (fabs(dxb) < r && fabs(dyb) < r)
            // RTCFDPoint M = {(i + 0.5) * f->h ,(j + 0.5) * f->h };

            if (isInsidePolygon(P, RTCFDPoint({(i + 0.5) * f->h, (j + 0.5) * f->h})))
            {
                f->s[i * n + j] = 0.0;
                if (this->RegionNr == 2)
                    f->m[i * n + j] = 0.5 + 0.5 * sin(0.1 * this->frameNr);
                else
                    f->m[i * n + j] = 1.0;

                f->u[i * n + j] = vx;
                f->u[(i + 1) * n + j] = vx;
                f->v[i * n + j] = vy;
                f->v[i * n + j + 1] = vy;
            }
        }
    }

    this->showObstacle = true;
    if (f) f->syncToTensors();
}

void Region::setObstacleNaca(double x, double y, bool reset)
{
    double vx = 0.0;
    double vy = 0.0;

    if (!reset)
    {
        vx = (x - this->obstacleX) / this->dt;
        vy = (y - this->obstacleY) / this->dt;
    }

    this->obstacleX = x;
    this->obstacleY = y;
    double r = this->characteristic_length;
    shared_ptr<Fluid> f = this->fluid;
    int n = f->numY;
    RTCFDPoint center = {x, y};
    vector<RTCFDPoint> P = getNacaPoints(center, r);
    //    double cd = sqrt(2) * f->h;
#pragma omp parallel for schedule(static) num_threads(f->numThreads)
    for (int i = 1; i < f->numX - 2; i++)
    {
        for (int j = 1; j < f->numY - 2; j++)
        {

            f->s[i * n + j] = 1.0;

            // double dx = (i + 0.5) * f->h - x;
            // double dy = (j + 0.5) * f->h - y;

            // //! axis change by a rotation of theta=pi/4
            // double dxb = sqrt(2) / 2 * (dx + dy);
            // double dyb = sqrt(2) / 2 * (-dx + dy);

            // if (fabs(dxb) < r && fabs(dyb) < r)
            // RTCFDPoint M = {(i + 0.5) * f->h ,(j + 0.5) * f->h };

            if (isInsidePolygon(P, RTCFDPoint({(i + 0.5) * f->h, (j + 0.5) * f->h})))
            {
                f->s[i * n + j] = 0.0;
                if (this->RegionNr == 2)
                    f->m[i * n + j] = 0.5 + 0.5 * sin(0.1 * this->frameNr);
                else
                    f->m[i * n + j] = 1.0;

                f->u[i * n + j] = vx;
                f->u[(i + 1) * n + j] = vx;
                f->v[i * n + j] = vy;
                f->v[i * n + j + 1] = vy;
            }
        }
    }

    this->showObstacle = true;
    if (f) f->syncToTensors();
}

void Region::setObstacleRotor(double x, double y, bool reset)
{
}

void Region::updateRegionSize(int _height, int _width)
{

    height = _height;
    width = _width;
    cScale = height / simHeight;
    simWidth = width / cScale;

// #if (DEBUG)
//     cout << "----------------" << endl;
//     cout << "updateRegionSize" << endl;
//     cout << "----------------" << endl;
//     cout << "height = " << height << endl;
//     cout << "width = " << width << endl;
//     cout << "simHeight = " << simHeight << endl;
//     cout << "cScale = " << cScale << endl;
//     cout << "simWidth = " << simWidth << endl;
// #endif
}

void Region::update()
{
    if (!paused && fluid) {
        fluid->simulate(dt, gravity, numIters, correctionStep);
        simulationSteps++;
        if (simulationSteps % 40 == 0 && simulationSteps <= 120) {
            std::string filename = "cpp_fields_" + std::to_string(simulationSteps) + ".txt";
            fluid->saveFields(filename);
            std::cout << "DEBUG: Saved fields at step " << simulationSteps << "." << std::endl;
        }
    }
}

void Region::loadDevelopedState(const std::string& filename) {
    if (!fluid) return;
    
    std::ifstream file(filename);
    if (!file.is_open()) {
        // Try parent directory if build folder is the current WD
        file.open("../" + filename);
    }
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open " << filename << " in . or .." << std::endl;
        return;
    }
    
    std::cout << "[REGION] Loading developed state from " << filename << "..." << std::endl;
    for (int i = 0; i < fluid->numX; i++) {
        for (int j = 0; j < fluid->numY; j++) {
            int idx = i * fluid->numY + j;
            if (!(file >> fluid->u[idx] >> fluid->v[idx] >> fluid->m[idx])) {
                std::cerr << "Error: Unexpected end of file in " << filename << std::endl;
                return;
            }
        }
    }
    // Sync corrected buffers
    memcpy(fluid->u_corrected.begin(), fluid->u.begin(), fluid->numCells * sizeof(float));
    memcpy(fluid->v_corrected.begin(), fluid->v.begin(), fluid->numCells * sizeof(float));
    fluid->cnt = 0;
    this->frameNr = 0;
    this->simulationSteps = 0; // Reset counter so comparison starts NOW
    std::cout << "[REGION] State loaded. Simulation counter reset." << std::endl;
    
    fluid->syncToTensors();
}



OBJ indexToOBJ(int index)
{
    switch (index)
    {
    case 0:
        return CYLINDER;
    case 1:
        return SQUARE;
    case 2:
        return DIAMOND;
    case 3:
        return NACA;
    case 4:
        return ROTOR;
    default:
        return CYLINDER;
    }
}
