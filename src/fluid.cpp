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
    numX = _numX + 2;
    numY = _numY + 2;
    numCells = numX * numY;
    h = _h;
    u.resize(numCells);
    v.resize(numCells);
    newU.resize(numCells);
    newV.resize(numCells);
    Vel.resize(numCells);
    p.resize(numCells);
    s.resize(numCells);
    m.resize(numCells, 1.0);
    newM.resize(numCells);
    u_corrected.resize(numX * numY, 0.0);
    v_corrected.resize(numX * numY, 0.0);
    num = numX * numY;
    cnt = 0;
    overRelaxation = _overRelaxation;
    numThreads = _numThreads;
}

void Fluid::integrate(double dt, double gravity)
{
    int n = numY;
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 1; i < numX; i++)
    {
        for (int j = 1; j < numY - 1; j++)
        {
            if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0)
#pragma omp atomic update
                v[i * n + j] += gravity * dt;
        }
    }
                            // std::exit(1); 
}

void Fluid::solveIncompressibility(int numIters, double dt)
{
    int n = numY;
    double cp = density * h * h / dt;
    
    for (int iter = 0; iter < numIters; iter++)
    {
#pragma omp parallel for schedule(static) num_threads(numThreads)
        for (int i = 1; i < numX - 1; i++)
        {
            for (int j = 1; j < numY - 1; j++)
            {
                if (s[i * n + j] == 0.0)
                // {                    std::cout << "Value of i: " << i<< "Value of j: " << j << std::endl;
                //             std::exit(1); }
                    continue;

                //                double s_ = s[i * n + j];
                double sx0 = s[(i - 1) * n + j];
                double sx1 = s[(i + 1) * n + j];
                double sy0 = s[i * n + j - 1];
                double sy1 = s[i * n + j + 1];
                double _s = sx0 + sx1 + sy0 + sy1;

                if (_s == 0.0)
                    continue;

                double div = u[(i + 1) * n + j] - u[i * n + j] + v[i * n + j + 1] - v[i * n + j];

                double _p = -div / _s;
#pragma omp atomic update
                _p *= overRelaxation;
#pragma omp atomic update
                p[i * n + j] += cp * _p;
#pragma omp atomic update
                u[i * n + j] -= sx0 * _p;
#pragma omp atomic update
                u[(i + 1) * n + j] += sx1 * _p;
#pragma omp atomic update
                v[i * n + j] -= sy0 * _p;
#pragma omp atomic update
                v[i * n + j + 1] += sy1 * _p;
            }
        }
    }
}

void Fluid::extrapolate()
{
    int n = numY;
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 0; i < numX; i++)
    {
        u[i * n + 0] = u[i * n + 1];
        u[i * n + numY - 1] = u[i * n + numY - 2];
    }
#pragma omp parallel for schedule(static) num_threads(numThreads)
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
#pragma omp parallel for schedule(static) num_threads(numThreads)
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

    cnt= 0;
    tt= tt+dt;

    int n = numY;
    double h2 = 0.5 * h;
        double eps = std::numeric_limits<double>::epsilon();

// #pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 1; i < numX; i++)
    {
        for (int j = 1; j < numY; j++)
        {
// #pragma omp atomic update
            // cnt++;

            // if (std::abs(s[i * n + j]) < eps && std::abs(s[(i - 1) * n + j]) <  eps){

            //     cout<<" y:" << j * h + h2<<" x:"<<i * h + h2<< " i:" << i<< " j:" << j << std::endl;
            // }

            // u component
            if (s[i * n + j] != 0.0 && s[(i - 1) * n + j] != 0.0 && j < numY - 1)
            {
                double x = i * h;
                double y = j * h + h2;
                double _u = u[i * n + j];
                double _v = avgV(i, j);

#pragma omp atomic update
                x -= dt * _u;
#pragma omp atomic update
                y -= dt * _v;
                _u = sampleField(x, y, U_FIELD);
                newU[i * n + j] = _u;
            }
            // v component
            if (s[i * n + j] != 0.0 && s[i * n + j - 1] != 0.0 && i < numX - 1)
            {
                double x = i * h + h2;
                double y = j * h;
                double _u = avgU(i, j);
                double _v = v[i * n + j];
                double xx = 0 , yy = 0;

                xx = x;
                yy = y;
#pragma omp atomic update
                x -= dt * _u;
#pragma omp atomic update
                y -= dt * _v;
                _v = sampleField(x, y, V_FIELD);
                newV[i * n + j] = _v;

// std::ofstream fout;

//             if (std::abs(tt - dt * 10.0) < 1e-6) 
//                 {
//                     #pragma omp critical

//                  fout.open("/Users/mohammadmoezzibadi/Desktop/wxRTCFD2_correct/build/Vel_" + std::to_string(tt) + ".txt", ios::app);
 
//                      fout<<xx<< std::setw(10) << yy<<std::setw(82)<< newV[i * n + j]<<std::setw(82)<< newU[i * n + j]<< std::setw(82)<< endl;//_p<<std::setw(82)<<dt*cnt <<endl;
//                 }
//                  fout.close();
            }
        }

    }
            // std::exit(1); 

    u = newU;
    v = newV;

}


void Fluid::advectTracer(double dt)
{
    newM = m;

    int n = numY;
    double h2 = 0.5 * h;
#pragma omp parallel for schedule(static) num_threads(numThreads)
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
void Fluid::applyCorrection(torch::jit::script::Module& model, double inVel)
{
    double dt = 1.0 / 60.0;
    // ttt += dt;
    double h2 = 0.5 * h;

    int Ny = numY;
    int Nx = numX;

    // Convert staggered u,v,m to 2D tensors
    torch::Tensor m_2d = torch::from_blob(m.data(), {Ny, Nx}, torch::kDouble).clone();
    torch::Tensor u_2d = torch::zeros({Ny, Nx - 1}, torch::kDouble);
    torch::Tensor v_2d = torch::zeros({Ny - 1, Nx}, torch::kDouble);

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            int idx = i * Ny + j;
            if (i < Nx - 1) u_2d[j][i] = u[idx];
            if (j < Ny - 1) v_2d[j][i] = v[idx];

            // Optional debug output
            // if (std::abs(ttt - dt * 10.0) < 1e-6) {
            //     double x = i * h + h2;
            //     double y = j * h;
            //     #pragma omp critical
            //     {
            //         std::ofstream fout("/Users/mohammadmoezzibadi/Desktop/wxRTCFD2_correct/build/VelCorrdfd_" 
            //                             + std::to_string(ttt) + ".txt", ios::app);
            //         fout << x << std::setw(10) << y
            //              << std::setw(20) << v[idx]
            //              << std::setw(20) << u[idx]
            //              << std::endl;
            //     }
            // }
        }
    }

    // --- 2. Pad u and v to match network input size ---
    auto u_pad = torch::constant_pad_nd(u_2d, {0, 1, 0, 0}, 0.0); // pad last column
    auto v_pad = torch::constant_pad_nd(v_2d, {0, 0, 0, 1}, 0.0); // pad last row

    // Normalize inlet velocity (if network is trained for different inVel)
    const double DATA_RE_MEAN = 1237.79296875;
    const double DATA_RE_STD  = 1453.7359614526729;
    double inVel_norm = (inVel - DATA_RE_MEAN) / DATA_RE_STD;

    // Stack into [1,3,Ny,Nx] tensor
    torch::Tensor net_in = torch::stack({m_2d, u_pad, v_pad}, 0)
                                .unsqueeze(0)
                                .to(torch::kFloat32);

    // Forward pass
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(net_in);
    torch::Tensor net_out = model.forward(inputs).toTensor();

    torch::Tensor cx = net_out.index({0, 0}); // Delta u corrections
    torch::Tensor cy = net_out.index({0, 1}); // Delta v corrections

    // Align corrections with staggered grid
    auto u_corr = cx.index({torch::indexing::Slice(), torch::indexing::Slice(c10::nullopt, -1)}).to(torch::kDouble).cpu();
    auto v_corr = cy.index({torch::indexing::Slice(c10::nullopt, -1), torch::indexing::Slice()}).to(torch::kDouble).cpu();

    torch::Tensor u_corrected_tensor = (u_2d + u_corr).to(torch::kDouble).cpu();
    torch::Tensor v_corrected_tensor = (v_2d + v_corr).to(torch::kDouble).cpu();

    // Flatten (std::vector<double>)
    u_corrected.resize(Nx * Ny, 0.0);
    v_corrected.resize(Nx * Ny, 0.0);

    auto u_acc = u_corrected_tensor.accessor<double,2>();
    auto v_acc = v_corrected_tensor.accessor<double,2>();

    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            if (i < Nx - 1)
                u_corrected[i * Ny + j] = u_acc[j][i];
            if (j < Ny - 1)
                v_corrected[i * Ny + j] = v_acc[j][i];
        }
    }

    // Optional debug print for first few elements
    // print_first_n(u_corrected_tensor, 10, "u_corrected_tensor");
    // print_first_n(v_corrected_tensor, 10, "v_corrected_tensor");
}


void Fluid::simulate(double dt, double gravity, int numIters,
                     const std::function<void(Fluid&)>& correctionStep){
    integrate(dt, gravity);
    fill(p.begin(), p.end(), 0.0);
    solveIncompressibility(numIters, dt);
    extrapolate();
    advectVelocity(dt);
    correctionStep(*this);
    advectTracer(dt);
    computeVelosityMagnitude();
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
