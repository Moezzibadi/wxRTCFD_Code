#include "fluid.h"
#include<iostream>       // std::cout, std::endl
#include <fstream>
#include <iomanip>      // std::setw
#include <iostream>
#include <c10/util/Optional.h>

#ifdef N_
#undef N_
#endif

#include <torch/torch.h>
#include <torch/script.h>

Fluid::Fluid(double _density, int _numX, int _numY, double _h, double _overRelaxation, int _numThreads)
{
    density = _density;
    numX = _numX;
    numY = _numY;
    numCells = numX * numY;
    h = _h;
    std::cout << "[FLUID] Constructor: numX = " << numX << ", numY = " << numY << ", h = " << h << std::endl;
    u.resize(numCells, 0.0);
    v.resize(numCells, 0.0);
    newU.resize(numCells, 0.0);
    newV.resize(numCells, 0.0);
    Vel.resize(numCells, 0.0);
    p.resize(numCells, 0.0);
    s.resize(numCells, 1.0);
    m.resize(numCells, 1.0);
    newM.resize(numCells, 1.0);
    u_corrected.resize(numCells, 0.0);
    v_corrected.resize(numCells, 0.0);
    num = numX * numY;
    cnt = 0;
    overRelaxation = _overRelaxation;
    numThreads = _numThreads;
    mlCorrectionEnabled = false;
}
void Fluid::saveFields(const std::string& filename) {
    std::ofstream f(filename);
    f << std::setprecision(18) << std::scientific;
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < numY; j++) {
            int idx = i * numY + j;
            f << u[idx] << " " << v[idx] << " " << m[idx] << "\n";
        }
    }
    f.close();
    std::cout << "Fields saved to " << filename << std::endl;
}

void Fluid::integrate(double dt, double gravity)
{
    int n = numY;
#pragma omp parallel for schedule(static) num_threads(numThreads) if(!mlCorrectionEnabled)
    for (int i = 1; i < numX; i++)
    {
        for (int j = 1; j < numY - 1; j++)
        {
            if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0)
#pragma omp atomic update
                v[i * n + j] += gravity * dt;
        }
    }
}


void Fluid::solveIncompressibility(int numIters, double dt)
{
    int n = numY;
    double cp = density * h * h / dt;
    
    for (int iter = 0; iter < numIters; iter++)
    {
        for (int i = 1; i < numX - 1; i++)
        {
            for (int j = 1; j < numY - 1; j++)
            {
                int idc = i * n + j;
                if (s[idc] == 0.0) continue;

                double sx0 = s[(i - 1) * n + j];
                double sx1 = s[(i + 1) * n + j];
                double sy0 = s[i * n + j - 1];
                double sy1 = s[i * n + j + 1];
                double _s = sx0 + sx1 + sy0 + sy1;

                if (_s == 0.0) continue;

                double div = u[(i + 1) * n + j] - u[idc] + v[i * n + j + 1] - v[idc];

                double _p = -div / _s * overRelaxation;
                p[idc] += cp * _p;
                u[idc] -= sx0 * _p;
                u[(i + 1) * n + j] += sx1 * _p;
                v[idc] -= sy0 * _p;
                v[i * n + j + 1] += sy1 * _p;
            }
        }
    }
}

void Fluid::extrapolate()
{
    int n = numY;
#pragma omp parallel for schedule(static) num_threads(numThreads) if(!mlCorrectionEnabled)
    for (int i = 0; i < numX; i++)
    {
        u[i * n + 0] = u[i * n + 1];
        u[i * n + numY - 1] = u[i * n + numY - 2];
    }
#pragma omp parallel for schedule(static) num_threads(numThreads) if(!mlCorrectionEnabled)
    for (int j = 0; j < numY; j++)
    {
        v[0 * n + j] = v[1 * n + j];
        v[(numX - 1) * n + j] = v[(numX - 2) * n + j];
    }
}

double Fluid::sampleField(double x, double y, int field)
{
    int n = numY;
    double h1 = 1.0 / h;
    double h2 = 0.5 * h;

    x = fmax(fmin(x, numX * h), h);
    y = fmax(fmin(y, numY * h), h);

    double dx = 0.0;
    double dy = 0.0;

    vector<double> f;

    switch (field)
    {
    case U_FIELD:
        f = u;
        dy = h2;
        break;
    case V_FIELD:
        f = v;
        dx = h2;
        break;
    case S_FIELD:
        f = m;
        dx = h2;
        dy = h2;
        break;
    }

    double x0 = fmin(floor((x - dx) * h1), numX - 1);
    double tx = ((x - dx) - x0 * h) * h1;
    double x1 = fmin(x0 + 1, numX - 1);

    double y0 = fmin(floor((y - dy) * h1), numY - 1);
    double ty = ((y - dy) - y0 * h) * h1;
    double y1 = fmin(y0 + 1, numY - 1);

    double sx = 1.0 - tx;
    double sy = 1.0 - ty;

    double val = sx * sy * f[x0 * n + y0] +
                 tx * sy * f[x1 * n + y0] +
                 tx * ty * f[x1 * n + y1] +
                 sx * ty * f[x0 * n + y1];

    return val;
}

double Fluid::avgU(int i, int j)
{
    int n = numY;
    return (u[i * n + j - 1] + u[i * n + j] +
            u[(i + 1) * n + j - 1] + u[(i + 1) * n + j]) *
           0.25;
}

double Fluid::avgV(int i, int j)
{
    int n = numY;
    return (v[(i - 1) * n + j] + v[i * n + j] +
            v[(i - 1) * n + j + 1] + v[i * n + j + 1]) *
           0.25;
}

void Fluid::computeVelosityMagnitude()
{
    int n = numY;
#pragma omp parallel for schedule(static) num_threads(numThreads) if(!mlCorrectionEnabled)
    for (int i = 0; i < numX; i++)
    {
        for (int j = 0; j < numY; j++)
        {
            Vel[i * n + j] = sqrt(pow(u[i * n + j], 2) + pow(v[i * n + j], 2));
        }
    }
}

void Fluid::advectVelocity(double dt)
{
    newU = u;
    newV = v;
    int n = numY;
    double h2 = 0.5 * h;

#pragma omp parallel for schedule(static) num_threads(numThreads) if(!mlCorrectionEnabled)
    for (int i = 1; i < numX; i++)
    {
        for (int j = 1; j < numY; j++)
        {
            // u component
            if (s[i * n + j] != 0.0 && s[(i - 1) * n + j] != 0.0 && j < numY - 1)
            {
                double x = i * h;
                double y = j * h + h2;
                double _u = u[i * n + j];
                double _v = (v[(i - 1) * n + j] + v[i * n + j] + v[(i - 1) * n + j + 1] + v[i * n + j + 1]) * 0.25;
                x -= dt * _u;
                y -= dt * _v;
                newU[i * n + j] = sampleField(x, y, U_FIELD);
            }
            // v component
            if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0 && i < numX - 1)
            {
                double x = i * h + h2;
                double y = j * h;
                double _u = (u[i * n + j - 1] + u[i * n + j] + u[(i + 1) * n + j - 1] + u[(i + 1) * n + j]) * 0.25;
                double _v = v[i * n + j];
                x -= dt * _u;
                y -= dt * _v;
                newV[i * n + j] = sampleField(x, y, V_FIELD);
            }
        }
    }
    u = newU;
    v = newV;
}


void Fluid::advectTracer(double dt)
{
    newM = m;

    int n = numY;
    double h2 = 0.5 * h;
#pragma omp parallel for schedule(static) num_threads(numThreads) if(!mlCorrectionEnabled)
    for (int i = 1; i < numX - 1; i++)
    {
        for (int j = 1; j < numY - 1; j++)
        {

            if (s[i * n + j] != 0.0)
            {
                double _u = (u[i * n + j] + u[(i + 1) * n + j]) * 0.5;
                double _v = (v[i * n + j] + v[i * n + j + 1]) * 0.5;
                double x = i * h + h2 - dt * _u;
                double y = j * h + h2 - dt * _v;

                newM[i * n + j] = sampleField(x, y, S_FIELD);
            }
        }
    }
    m = newM;
}

// void print_first_n(const torch::Tensor& t, int n, const std::string& name) {
//     std::cout << name << ": ";
//     auto flat = t.flatten();  // flatten to 1D view
//     int size = std::min<int>(n, flat.size(0));
//     for (int i = 0; i < size; i++) {
//         std::cout << flat[i].item<double>() << " ";
//     }
//     std::cout << std::endl;
// }

void Fluid::NoCorrection()
{
    u_corrected = u;
    v_corrected = v;   
}
#include <torch/torch.h>
#include <torch/script.h>
#include <algorithm>

void Fluid::applyCorrection(torch::jit::script::Module& model, double inVel)
{
    torch::NoGradGuard no_grad;

    // Get the device the model is currently on
    torch::Device device = (*model.parameters().begin()).device();

    const double U_MEAN = 3.1703, U_STD = 1.9038;
    const double V_MEAN = -0.0519, V_STD = 1.4366;
    const double DATA_RE_MEAN = 3.5416666666666665, DATA_RE_STD = 1.2891680903122622;
    const double MAX_CORR = 0.5;

    int Ny_solver = numY, Nx_solver = numX;
    const int Ny_net = 52, Nx_net = 88;

    // 1. Zero-copy: Wrap C++ vectors into Tensors
    // Note: our vectors are (Nx * Ny), we view them as (Nx, Ny)
    auto options = torch::TensorOptions().dtype(torch::kFloat64);
    torch::Tensor u_ts = torch::from_blob(u.data(), {Nx_solver, Ny_solver}, options).to(torch::kFloat32);
    torch::Tensor v_ts = torch::from_blob(v.data(), {Nx_solver, Ny_solver}, options).to(torch::kFloat32);

    // Normalize and prepare channels
    float inVel_norm = (DATA_RE_STD > 1e-6) ? (float)((inVel - DATA_RE_MEAN) / DATA_RE_STD) : 0.0f;
    
    torch::Tensor c1 = torch::full({1, 1, Ny_solver, Nx_solver}, inVel_norm);
    torch::Tensor c2 = ((u_ts.transpose(0, 1) - U_MEAN) / U_STD).unsqueeze(0).unsqueeze(0);
    torch::Tensor c3 = ((v_ts.transpose(0, 1) - V_MEAN) / V_STD).unsqueeze(0).unsqueeze(0);
    
    torch::Tensor net_in = torch::cat({c1, c2, c3}, 1);

    // 2. Resize ONLY if necessary
    torch::Tensor net_in_resized;
    if (Ny_solver == Ny_net && Nx_solver == Nx_net) {
        net_in_resized = net_in.to(device);
    } else {
        net_in_resized = torch::upsample_bilinear2d(net_in.to(device), {Ny_net, Nx_net}, false);
    }

    // 3. Forward pass
    torch::Tensor net_out;
    try {
        net_out = model.forward({net_in_resized}).toTensor().to(torch::kCPU);
    } catch (const std::exception& e) {
        std::cerr << "ML Inference failed: " << e.what() << std::endl;
        NoCorrection();
        return;
    }

    // 4. Resize back ONLY if necessary
    torch::Tensor net_out_resized;
    if (Ny_solver == Ny_net && Nx_solver == Nx_net) {
        net_out_resized = net_out;
    } else {
        net_out_resized = torch::upsample_bilinear2d(net_out, {Ny_solver, Nx_solver}, false);
    }

    // 5. Apply correction using optimized Tensor operations
    net_out_resized = net_out_resized.clamp(-MAX_CORR, MAX_CORR).to(torch::kFloat64);
    auto out_acc = net_out_resized.accessor<double, 4>();

    // Final application loop (Boundary checks still required)
    for (int i = 0; i < Nx_solver; i++) {
        for (int j = 0; j < Ny_solver; j++) {
            int idx = i * Ny_solver + j;
            
            double cx = out_acc[0][0][j][i];
            double cy = out_acc[0][1][j][i];

            if (i > 0 && i < Nx_solver - 1 && s[idx] != 0.0 && s[(i-1) * Ny_solver + j] != 0.0) {
                u[idx] += cx;
            }
            if (j > 0 && j < Ny_solver - 1 && s[idx] != 0.0 && s[idx - 1] != 0.0) {
                v[idx] += cy;
            }
            
            u_corrected[idx] = u[idx];
            v_corrected[idx] = v[idx];
        }
    }
}







void Fluid::simulate(double dt, double gravity, int numIters,
                     const std::function<void(Fluid&)>& correctionStep){
    integrate(dt, gravity);
    fill(p.begin(), p.end(), 0.0);
    solveIncompressibility(numIters, dt);
    extrapolate();
    advectVelocity(dt);
    advectTracer(dt);
    correctionStep(*this);
    computeVelosityMagnitude();
    tt += dt;
}

void Fluid::updateFluidParameters()
{
    numCells = numX * numY;
    u.resize(numCells);
    v.resize(numCells);
    // tt = tt;
    newU.resize(numCells);
    newV.resize(numCells);
    p.resize(numCells);
    s.resize(numCells);
    m.resize(numCells, 1.0);
    newM.resize(numCells);
    num = numX * numY;
}
