#include "fluid.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <c10/util/Optional.h>
#include <mach-o/dyld.h>
#include <chrono>
#include <mutex>

#include <torch/torch.h>
#include <torch/script.h>

#ifdef __APPLE__
#import <Metal/Metal.h>
#import <QuartzCore/CAMetalLayer.h>

// Internal struct to hold Objective-C Metal objects
struct MetalState {
    id<MTLDevice> device;
    id<MTLCommandQueue> commandQueue;
    id<MTLComputePipelineState> pipelineState;
    id<MTLComputePipelineState> integrateState;
    id<MTLComputePipelineState> extrapolateState;
    
    // GPU Buffers
    id<MTLBuffer> uBuffer;
    id<MTLBuffer> vBuffer;
    id<MTLBuffer> sBuffer;
    id<MTLBuffer> pBuffer;
};
#endif

Fluid::Fluid(float _density, int _numX, int _numY, float _h, float _overRelaxation, int _numThreads)
{
    density = _density;
    numX = _numX;
    numY = _numY;
    numCells = numX * numY;
    h = _h;
    std::cout << "[FLUID] Constructor (FP32): numX = " << numX << ", numY = " << numY << ", h = " << h << std::endl;
    
    u.resize(numCells);
    v.resize(numCells);
    newU.resize(numCells);
    newV.resize(numCells);
    Vel.resize(numCells);
    p.resize(numCells);
    s.resize(numCells);
    std::fill(s.begin(), s.end(), 1.0f);
    m.resize(numCells);
    std::fill(m.begin(), m.end(), 1.0f);
    newM.resize(numCells);
    std::fill(newM.begin(), newM.end(), 1.0f);
    u_corrected.resize(numCells);
    v_corrected.resize(numCells);
    
    overRelaxation = _overRelaxation;
    numThreads = _numThreads;

    // --- GPU / Metal Initialization ---
#ifdef __APPLE__
    @autoreleasepool {
        id<MTLDevice> mtlDevice = MTLCreateSystemDefaultDevice();
        if (mtlDevice) {
            std::cout << "[FLUID] Direct Metal hardware access ENABLED: " << [mtlDevice.name UTF8String] << std::endl;
            
            metal = std::make_unique<MetalState>();
            metal->device = mtlDevice;
            metal->commandQueue = [metal->device newCommandQueue];
            
            // JIT compile the Metal shader from source (FP32 VERSION)
            NSError* error = nil;
            NSString* shaderSource = @"#include <metal_stdlib>\n"
                                     "using namespace metal;\n"
                                     "kernel void solveIncompressibility(\n"
                                     "    device float* u [[buffer(0)]],\n"
                                     "    device float* v [[buffer(1)]],\n"
                                     "    device const float* s [[buffer(2)]],\n"
                                     "    device float* p [[buffer(3)]],\n"
                                     "    constant int& numX [[buffer(4)]],\n"
                                     "    constant int& numY [[buffer(5)]],\n"
                                     "    constant float& cp [[buffer(6)]],\n"
                                     "    constant float& overRelaxation [[buffer(7)]],\n"
                                     "    constant int& pass [[buffer(10)]],\n"
                                     "    uint2 id [[thread_position_in_grid]])\n"
                                     "{\n"
                                     "    if (id.x < 1 || id.x >= (uint)numX - 1 || id.y < 1 || id.y >= (uint)numY - 1) return;\n"
                                     "    if ((int(id.x + id.y) % 2) != pass) return;\n"
                                     "    int n = numY;\n"
                                     "    int idc = id.x * n + id.y;\n"
                                     "    if (s[idc] == 0.0f) return;\n"
                                     "    float sx0 = s[(id.x - 1) * n + id.y];\n"
                                     "    float sx1 = s[(id.x + 1) * n + id.y];\n"
                                     "    float sy0 = s[id.x * n + id.y - 1];\n"
                                     "    float sy1 = s[id.x * n + id.y + 1];\n"
                                     "    float _s = sx0 + sx1 + sy0 + sy1;\n"
                                     "    if (_s == 0.0f) return;\n"
                                     "    float div = u[(id.x + 1) * n + id.y] - u[idc] + v[id.x * n + id.y + 1] - v[idc];\n"
                                     "    float _p = -div / _s * overRelaxation;\n"
                                     "    p[idc] += cp * _p;\n"
                                     "    u[idc] -= sx0 * _p;\n"
                                     "    u[(id.x + 1) * n + id.y] += sx1 * _p;\n"
                                     "    v[idc] -= sy0 * _p;\n"
                                     "    v[id.x * n + id.y + 1] += sy1 * _p;\n"
                                     "}\n"
                                     "kernel void integrate(\n"
                                     "    device float* v [[buffer(1)]],\n"
                                     "    device const float* s [[buffer(2)]],\n"
                                     "    constant int& numX [[buffer(4)]],\n"
                                     "    constant int& numY [[buffer(5)]],\n"
                                     "    constant float& dt [[buffer(8)]],\n"
                                     "    constant float& gravity [[buffer(9)]],\n"
                                     "    uint2 id [[thread_position_in_grid]])\n"
                                     "{\n"
                                     "    if (id.x < 1 || id.x >= (uint)numX || id.y < 1 || id.y >= (uint)numY - 1) return;\n"
                                     "    int n = numY;\n"
                                     "    if (s[id.x * n + id.y] != 0.0f && s[id.x * n + id.y - 1] != 0.0f) {\n"
                                     "        v[id.x * n + id.y] += gravity * dt;\n"
                                     "    }\n"
                                     "}\n"
                                     "kernel void extrapolate(\n"
                                     "    device float* u [[buffer(0)]],\n"
                                     "    device float* v [[buffer(1)]],\n"
                                     "    constant int& numX [[buffer(4)]],\n"
                                     "    constant int& numY [[buffer(5)]],\n"
                                     "    uint2 id [[thread_position_in_grid]])\n"
                                     "{\n"
                                     "    int n = numY;\n"
                                     "    // Only run on specific edge threads to avoid race conditions\n"
                                     "    if (id.x < (uint)numX && id.y == 0) {\n"
                                     "        u[id.x * n + 0] = u[id.x * n + 1];\n"
                                     "        u[id.x * n + n - 1] = u[id.x * n + n - 2];\n"
                                     "    }\n"
                                     "    if (id.y < (uint)numY && id.x == 0) {\n"
                                     "        v[0 * n + id.y] = v[1 * n + id.y];\n"
                                     "        v[(numX - 1) * n + id.y] = v[(numX - 2) * n + id.y];\n"
                                     "    }\n"
                                     "}";
            
            id<MTLLibrary> library = [metal->device newLibraryWithSource:shaderSource options:nil error:&error];
            
            if (library) {
                metal->pipelineState = [metal->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"solveIncompressibility"] error:&error];
                metal->integrateState = [metal->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"integrate"] error:&error];
                metal->extrapolateState = [metal->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"extrapolate"] error:&error];
                
                if (metal->pipelineState && metal->integrateState && metal->extrapolateState) {
                    // Create persistent buffers using NoCopy (Zero-copy Shared memory)
                    // PaddedSize is essential for Metal newBufferWithBytesNoCopy (must be multiple of 4096)
                    metal->uBuffer = [metal->device newBufferWithBytesNoCopy:u.begin() length:u.paddedSize options:MTLResourceStorageModeShared deallocator:nil];
                    metal->vBuffer = [metal->device newBufferWithBytesNoCopy:v.begin() length:v.paddedSize options:MTLResourceStorageModeShared deallocator:nil];
                    metal->sBuffer = [metal->device newBufferWithBytesNoCopy:s.begin() length:s.paddedSize options:MTLResourceStorageModeShared deallocator:nil];
                    metal->pBuffer = [metal->device newBufferWithBytesNoCopy:p.begin() length:p.paddedSize options:MTLResourceStorageModeShared deallocator:nil];
                    
                    if (metal->uBuffer && metal->vBuffer && metal->sBuffer && metal->pBuffer) {
                        useMetal = true;
                        std::cout << "[FLUID] Metal Zero-Copy Shared Memory initialized (Padded: " << u.paddedSize << " bytes)." << std::endl;
                    } else {
                        std::cerr << "[FLUID] Metal Buffer Allocation Error (Size misalignment?)" << std::endl;
                        useMetal = false;
                    }
                } else {
                    std::cerr << "[FLUID] Metal Pipeline Error: " << [[error localizedDescription] UTF8String] << std::endl;
                }
            } else {
                std::cerr << "[FLUID] Metal JIT Compilation Error: " << [[error localizedDescription] UTF8String] << std::endl;
            }
        }
    }
#endif

    // LibTorch device detection (FORCED CPU for main state to allow seamless sync with vectors/Metal)
    device = torch::kCPU;
    std::cout << "[FLUID] LibTorch (State) on CPU for zero-copy sync with Metal" << std::endl;

    // Initialize Tensors (FP32)
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    u_ts = torch::from_blob(u.begin(), {numX, numY}, options);
    v_ts = torch::from_blob(v.begin(), {numX, numY}, options);
    p_ts = torch::from_blob(p.begin(), {numX, numY}, options);
    s_ts = torch::from_blob(s.begin(), {numX, numY}, options);
    m_ts = torch::from_blob(m.begin(), {numX, numY}, options);
    
    // Auxiliary tensors still on CPU
    newU_ts = torch::zeros({numX, numY}, options);
    newV_ts = torch::zeros({numX, numY}, options);
    newM_ts = torch::ones({numX, numY}, options);
    
    // Step 5: Persistent ML Input & Correction Tensors
    net_in_ts = torch::empty({1, 3, numY, numX}, options);
    last_cx = torch::zeros({numX, numY}, options);
    last_cy = torch::zeros({numX, numY}, options);
    cnt = 0;
    m_mlBusy = false;
    m_hasFreshData = false;
    m_stepsRemaining = 0;
}

Fluid::~Fluid() {
    if (m_mlThread.joinable()) {
        m_mlThread.join();
    }
    // Metal objects are cleaned up by unique_ptr and ARC
}

void Fluid::syncToTensors() {
    torch::NoGradGuard no_grad;
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    u_ts = torch::from_blob(u.begin(), {numX, numY}, options);
    v_ts = torch::from_blob(v.begin(), {numX, numY}, options);
    p_ts = torch::from_blob(p.begin(), {numX, numY}, options);
    s_ts = torch::from_blob(s.begin(), {numX, numY}, options);
    m_ts = torch::from_blob(m.begin(), {numX, numY}, options);
    
    // Step 5: Re-initialize persistent tensors on resize
    net_in_ts = torch::empty({1, 3, numY, numX}, options);
    last_cx = torch::zeros({numX, numY}, options);
    last_cy = torch::zeros({numX, numY}, options);
}

void Fluid::syncToVectors() {
    // No-op because u_ts uses from_blob(u.begin()) on CPU
}

void Fluid::solveIncompressibilityMetal(int numIters, float dt) {
#ifdef __APPLE__
    if (!useMetal) return;

    @autoreleasepool {
        float cp = density * h * h / dt;
        MTLSize threadsPerGrid = MTLSizeMake(numX, numY, 1);
        NSUInteger w = metal->pipelineState.threadExecutionWidth;
        NSUInteger h_th = metal->pipelineState.maxTotalThreadsPerThreadgroup / w;
        MTLSize threadsPerThreadgroup = MTLSizeMake(w, h_th, 1);

        for (int iter = 0; iter < numIters; iter++) {
            for (int pass = 0; pass < 2; pass++) {
                id<MTLCommandBuffer> commandBuffer = [metal->commandQueue commandBuffer];
                id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

                [computeEncoder setComputePipelineState:metal->pipelineState];
                [computeEncoder setBuffer:metal->uBuffer offset:0 atIndex:0];
                [computeEncoder setBuffer:metal->vBuffer offset:0 atIndex:1];
                [computeEncoder setBuffer:metal->sBuffer offset:0 atIndex:2];
                [computeEncoder setBuffer:metal->pBuffer offset:0 atIndex:3];
                [computeEncoder setBytes:&numX length:sizeof(int) atIndex:4];
                [computeEncoder setBytes:&numY length:sizeof(int) atIndex:5];
                [computeEncoder setBytes:&cp length:sizeof(float) atIndex:6];
                [computeEncoder setBytes:&overRelaxation length:sizeof(float) atIndex:7];
                [computeEncoder setBytes:&pass length:sizeof(int) atIndex:10];

                [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:threadsPerThreadgroup];
                [computeEncoder endEncoding];
                [commandBuffer commit];
                [commandBuffer waitUntilCompleted];
            }
        }
    }
#endif
}
void Fluid::saveFields(const std::string& filename) {
    std::ofstream f(filename);
    f << std::setprecision(18) << std::scientific;
    float* u_ptr = u.begin();
    float* v_ptr = v.begin();
    float* m_ptr = m.begin();
    for (int i = 0; i < numX; i++) {
        for (int j = 0; j < numY; j++) {
            int idx = i * numY + j;
            f << u_ptr[idx] << " " << v_ptr[idx] << " " << m_ptr[idx] << "\n";
        }
    }
    f.close();
    std::cout << "Fields saved to " << filename << std::endl;
}

void Fluid::integrate(float dt, float gravity)
{
    if (useMetal) {
        integrateMetal(dt, gravity);
        return;
    }
    torch::NoGradGuard no_grad;
    auto s_curr = s_ts.slice(1, 1, numY - 1).slice(0, 1, numX);
    auto s_prev_y = s_ts.slice(1, 0, numY - 2).slice(0, 1, numX);
    auto mask = (s_curr != 0.0f) & (s_prev_y != 0.0f);
    v_ts.slice(1, 1, numY - 1).slice(0, 1, numX).masked_fill_(mask, v_ts.slice(1, 1, numY - 1).slice(0, 1, numX) + gravity * dt);
}

void Fluid::integrateMetal(float dt, float gravity) {
#ifdef __APPLE__
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [metal->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:metal->integrateState];
        [computeEncoder setBuffer:metal->vBuffer offset:0 atIndex:1];
        [computeEncoder setBuffer:metal->sBuffer offset:0 atIndex:2];
        [computeEncoder setBytes:&numX length:sizeof(int) atIndex:4];
        [computeEncoder setBytes:&numY length:sizeof(int) atIndex:5];
        [computeEncoder setBytes:&dt length:sizeof(float) atIndex:8];
        [computeEncoder setBytes:&gravity length:sizeof(float) atIndex:9];

        MTLSize threadsPerGrid = MTLSizeMake(numX, numY, 1);
        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
#endif
}

void Fluid::solveIncompressibility(int numIters, float dt)
{
    if (useMetal) {
        solveIncompressibilityMetal(numIters, dt);
        return;
    }

    torch::NoGradGuard no_grad;
    float cp = density * h * h / dt;
    p_ts.zero_();

    for (int iter = 0; iter < numIters; iter++)
    {
        auto u_curr = u_ts.slice(0, 1, numX - 1).slice(1, 1, numY - 1);
        auto u_next_x = u_ts.slice(0, 2, numX).slice(1, 1, numY - 1);
        auto v_curr = v_ts.slice(0, 1, numX - 1).slice(1, 1, numY - 1);
        auto v_next_y = v_ts.slice(0, 1, numX - 1).slice(1, 2, numY);
        auto div = u_next_x - u_curr + v_next_y - v_curr;
        auto sx0 = s_ts.slice(0, 0, numX - 2).slice(1, 1, numY - 1);
        auto sx1 = s_ts.slice(0, 2, numX).slice(1, 1, numY - 1);
        auto sy0 = s_ts.slice(0, 1, numX - 1).slice(1, 0, numY - 2);
        auto sy1 = s_ts.slice(0, 1, numX - 1).slice(1, 2, numY);
        auto s_sum = sx0 + sx1 + sy0 + sy1;
        auto mask = (s_ts.slice(0, 1, numX - 1).slice(1, 1, numY - 1) != 0.0f) & (s_sum > 0.0f);
        auto dp = torch::zeros_like(div);
        dp.masked_scatter_(mask, -div.masked_select(mask) / s_sum.masked_select(mask) * overRelaxation);
        p_ts.slice(0, 1, numX - 1).slice(1, 1, numY - 1) += cp * dp;
        u_ts.slice(0, 1, numX - 1).slice(1, 1, numY - 1) -= sx0 * dp;
        u_ts.slice(0, 2, numX).slice(1, 1, numY - 1) += sx1 * dp;
        v_ts.slice(0, 1, numX - 1).slice(1, 1, numY - 1) -= sy0 * dp;
        v_ts.slice(0, 1, numX - 1).slice(1, 2, numY) += sy1 * dp;
    }
}

void Fluid::extrapolate()
{
    if (useMetal) {
        extrapolateMetal();
        return;
    }
    torch::NoGradGuard no_grad;
    u_ts.select(1, 0) = u_ts.select(1, 1);
    u_ts.select(1, numY - 1) = u_ts.select(1, numY - 2);
    v_ts.select(0, 0) = v_ts.select(0, 1);
    v_ts.select(0, numX - 1) = v_ts.select(0, numX - 2);
}

void Fluid::extrapolateMetal() {
#ifdef __APPLE__
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = [metal->commandQueue commandBuffer];
        id<MTLComputeCommandEncoder> computeEncoder = [commandBuffer computeCommandEncoder];

        [computeEncoder setComputePipelineState:metal->extrapolateState];
        [computeEncoder setBuffer:metal->uBuffer offset:0 atIndex:0];
        [computeEncoder setBuffer:metal->vBuffer offset:0 atIndex:1];
        [computeEncoder setBytes:&numX length:sizeof(int) atIndex:4];
        [computeEncoder setBytes:&numY length:sizeof(int) atIndex:5];

        MTLSize threadsPerGrid = MTLSizeMake(numX, numY, 1);
        [computeEncoder dispatchThreads:threadsPerGrid threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [computeEncoder endEncoding];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
    }
#endif
}

float Fluid::avgU(int i, int j) {
    int n = numY;
    float* u_ptr = u.begin();
    int i0 = i, i1 = std::min(i + 1, numX - 1);
    int j0 = std::max(j - 1, 0), j1 = j;
    return (u_ptr[i0 * n + j0] + u_ptr[i1 * n + j0] + u_ptr[i0 * n + j1] + u_ptr[i1 * n + j1]) * 0.25f;
}

float Fluid::avgV(int i, int j) {
    int n = numY;
    float* v_ptr = v.begin();
    int i0 = std::max(i - 1, 0), i1 = i;
    int j0 = j, j1 = std::min(j + 1, numY - 1);
    return (v_ptr[i0 * n + j0] + v_ptr[i1 * n + j0] + v_ptr[i0 * n + j1] + v_ptr[i1 * n + j1]) * 0.25f;
}

float Fluid::sampleField(float x, float y, int field) {
    int n = numY;
    float h1 = 1.0f / h;
    float h2 = 0.5f * h;
    float dx = 0.0f, dy = 0.0f;
    const float* f = nullptr;

    if (field == 0) { f = u.begin(); dy = h2; }
    else if (field == 1) { f = v.begin(); dx = h2; }
    else { f = m.begin(); dx = h2; dy = h2; }

    float x_idx = fmax(fmin(x - dx, (numX - 1) * h), 0.0f) * h1;
    float y_idx = fmax(fmin(y - dy, (numY - 1) * h), 0.0f) * h1;

    int i0 = (int)floor(x_idx);
    float tx = x_idx - i0;
    int i1 = std::min(i0 + 1, numX - 1);
    int j0 = (int)floor(y_idx);
    float ty = y_idx - j0;
    int j1 = std::min(j0 + 1, numY - 1);

    float sx = 1.0f - tx, sy = 1.0f - ty;
    return sx * sy * f[i0 * n + j0] + tx * sy * f[i1 * n + j0] +
           tx * ty * f[i1 * n + j1] + sx * ty * f[i0 * n + j1];
}

void Fluid::computeVelosityMagnitude()
{
    torch::NoGradGuard no_grad;
    auto mag = torch::sqrt(u_ts.pow(2) + v_ts.pow(2));
    memcpy(Vel.begin(), mag.data_ptr(), numCells * sizeof(float));
}

void Fluid::advectVelocity(float dt)
{
    memcpy(newU.begin(), u.begin(), numCells * sizeof(float));
    memcpy(newV.begin(), v.begin(), numCells * sizeof(float));
    int n = numY; float h2 = 0.5f * h;
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 1; i < numX; i++) {
        for (int j = 1; j < numY; j++) {
            if (s[i * n + j] != 0.0f && s[(i - 1) * n + j] != 0.0f && j < numY - 1) {
                float x = i * h, y = j * h + h2;
                float _u = u[i * n + j], _v = avgV(i, j);
                x -= dt * _u; y -= dt * _v;
                newU[i * n + j] = sampleField(x, y, 0);
            }
            if (s[i * n + j] != 0.0f && s[i * n + j - 1] != 0.0f && i < numX - 1) {
                float x = i * h + h2, y = j * h;
                float _u = avgU(i, j), _v = v[i * n + j];
                x -= dt * _u; y -= dt * _v;
                newV[i * n + j] = sampleField(x, y, 1);
            }
        }
    }
    memcpy(u.begin(), newU.begin(), numCells * sizeof(float));
    memcpy(v.begin(), newV.begin(), numCells * sizeof(float));
}

void Fluid::advectTracer(float dt) {
    memcpy(newM.begin(), m.begin(), numCells * sizeof(float));
    int n = numY;
    float h2 = 0.5f * h;
#pragma omp parallel for schedule(static) num_threads(numThreads)
    for (int i = 1; i < numX - 1; i++) {
        for (int j = 1; j < numY - 1; j++) {
            if (s[i * n + j] != 0.0f) {
                float _u = (u[i * n + j] + u[(i + 1) * n + j]) * 0.5f;
                float _v = (v[i * n + j] + v[i * n + j + 1]) * 0.5f;
                float x = i * h + h2 - dt * _u;
                float y = j * h + h2 - dt * _v;
                newM[i * n + j] = sampleField(x, y, M_FIELD);
            }
        }
    }
    memcpy(m.begin(), newM.begin(), numCells * sizeof(float));
    
    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j < numY; j++) {
            float pipeH = 0.1f * numY;
            int minJ = (int)floor(0.5f * numY - 0.5f * pipeH);
            int maxJ = (int)floor(0.5f * numY + 0.5f * pipeH);
            if (j >= minJ && j < maxJ) {
                m[i * n + j] = 0.0f;
            } else {
                m[i * n + j] = 1.0f;
            }
        }
    }
}

void Fluid::NoCorrection() {
    memcpy(u_corrected.begin(), u.begin(), numCells * sizeof(float));
    memcpy(v_corrected.begin(), v.begin(), numCells * sizeof(float));
}

void Fluid::applyCorrection(torch::jit::script::Module& model, float inVel)
{
    torch::NoGradGuard no_grad;

    const float U_MEAN = 3.1703f, U_STD = 1.9038f;
    const float V_MEAN = -0.0519f, V_STD = 1.4366f;
    const float DATA_RE_MEAN = 3.5416666666666665f, DATA_RE_STD = 1.2891680903122622f;
    const float MAX_CORR = 1.5f; // INCREASED STRENGTH for better visibility

    int Ny_solver = numY, Nx_solver = numX;
    const int Ny_net = 52, Nx_net = 88;

    // KICK OFF ASYNC INFERENCE
    if (!m_mlBusy && m_stepsRemaining == 0) {
        float inVel_norm = (DATA_RE_STD > 1e-6f) ? (float)((inVel - DATA_RE_MEAN) / DATA_RE_STD) : 0.0f;

        torch::Tensor net_in = torch::empty({1, 3, Ny_solver, Nx_solver});
        net_in.select(1, 0).fill_(inVel_norm);
        net_in.select(1, 1).copy_(u_ts.transpose(0, 1).sub(U_MEAN).div(U_STD));
        net_in.select(1, 2).copy_(v_ts.transpose(0, 1).sub(V_MEAN).div(V_STD));

        m_mlBusy = true;
        if (m_mlThread.joinable()) m_mlThread.detach(); 
        
        m_mlThread = std::thread([this, &model, net_in, Ny_solver, Nx_solver, Ny_net, Nx_net, MAX_CORR]() {
            torch::NoGradGuard no_grad_bg;

            torch::Tensor net_in_resized;
            if (Ny_solver == Ny_net && Nx_solver == Nx_net) net_in_resized = net_in;
            else net_in_resized = torch::upsample_bilinear2d(net_in, {Ny_net, Nx_net}, false);

            torch::Tensor net_out = model.forward({net_in_resized}).toTensor();

            torch::Tensor net_out_resized;
            if (Ny_solver == Ny_net && Nx_solver == Nx_net) net_out_resized = net_out;
            else net_out_resized = torch::upsample_bilinear2d(net_out, {Ny_solver, Nx_solver}, false);

            net_out_resized.clamp_(-MAX_CORR, MAX_CORR);
            
            {
                std::lock_guard<std::mutex> lock(m_mlMutex);
                last_cx.copy_(net_out_resized.select(1, 0).squeeze(0).transpose(0, 1));
                last_cy.copy_(net_out_resized.select(1, 1).squeeze(0).transpose(0, 1));
                m_hasFreshData = true; 
            }
            m_mlBusy = false;
        });
    }

    if (m_hasFreshData) {
        m_hasFreshData = false;
        m_stepsRemaining = 1; // Apply in one burst but logged
        
        float u_mag = last_cx.abs().mean().item<float>();
        float v_mag = last_cy.abs().mean().item<float>();
        std::cout << "[ML LOG] Applied Correction. Mean U: " << u_mag << ", Mean V: " << v_mag << std::endl;
    }

    // Always copy for display
    memcpy(u_corrected.begin(), u.begin(), numCells * sizeof(float));
    memcpy(v_corrected.begin(), v.begin(), numCells * sizeof(float));
}


void Fluid::simulate(float dt, float gravity, int numIters,
                     const std::function<void(Fluid&)>& correctionStep){
    integrate(dt, gravity);
    
    // APPLY ML CORRECTION BEFORE SOLVER (Crucial!)
    // This makes the ML values part of the physical pressure calculation
    correctionStep(*this); 
    
    if (m_stepsRemaining > 0) {
        std::lock_guard<std::mutex> lock(m_mlMutex);
        if (numX > 1 && numY > 1) {
            auto u_target = u_ts.slice(0, 1, numX - 1);
            auto cx_slice = last_cx.slice(0, 1, numX - 1);
            auto s_mask_u = (s_ts.slice(0, 1, numX - 1) != 0.0f) & (s_ts.slice(0, 0, numX - 2) != 0.0f);
            u_target.add_(cx_slice * s_mask_u.to(torch::kFloat32));

            auto v_target = v_ts.slice(1, 1, numY - 1);
            auto cy_slice = last_cy.slice(1, 1, numY - 1);
            auto s_mask_v = (s_ts.slice(1, 1, numY - 1) != 0.0f) & (s_ts.slice(1, 0, numY - 2) != 0.0f);
            v_target.add_(cy_slice * s_mask_v.to(torch::kFloat32));
        }
        m_stepsRemaining = 0; // Reset
    }

    solveIncompressibility(numIters, dt);
    extrapolate();
    advectVelocity(dt);
    advectTracer(dt);
    computeVelosityMagnitude();
    tt += dt;
}

void Fluid::updateFluidParameters()
{
    numCells = numX * numY;
    u.resize(numCells);
    v.resize(numCells);
    newU.resize(numCells);
    newV.resize(numCells);
    p.resize(numCells);
    s.resize(numCells);
    m.resize(numCells);
    std::fill(m.begin(), m.end(), 1.0f);
    newM.resize(numCells);
    num = numX * numY;
    syncToTensors();
}
