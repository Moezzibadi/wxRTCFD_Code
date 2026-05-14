#ifndef FLUID_H
#define FLUID_H

#include <vector>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>

#ifdef N_
#define WX_N_DEFINED
#pragma push_macro("N_")
#undef N_
#endif

#include <torch/torch.h>
#include <torch/script.h>

#ifdef WX_N_DEFINED
#pragma pop_macro("N_")
#undef WX_N_DEFINED
#endif

// Field identifiers
enum {
    U_FIELD = 0,
    V_FIELD = 1,
    S_FIELD = 2,
    M_FIELD = 3
};

// Internal struct to hold Objective-C Metal objects
struct MetalState;

// Helper for shared memory management
struct SharedBuffer {
    float* data = nullptr;
    size_t size = 0;       // Number of elements
    size_t paddedSize = 0; // Padded size in bytes for Metal

    SharedBuffer() = default;
    
    // Disable copy for now to avoid double-free issues
    SharedBuffer(const SharedBuffer&) = delete;
    SharedBuffer& operator=(const SharedBuffer&) = delete;

    // Move is okay
    SharedBuffer(SharedBuffer&& other) noexcept : data(other.data), size(other.size), paddedSize(other.paddedSize) {
        other.data = nullptr;
        other.size = 0;
        other.paddedSize = 0;
    }

    ~SharedBuffer() {
        if (data) free(data);
    }

    void resize(size_t n) {
        if (data) free(data);
        size = n;
        size_t bytes = n * sizeof(float);
        // Metal requires bytes to be multiple of 4096 for NoCopy
        paddedSize = (bytes + 4095) & ~4095; 
        
        if (posix_memalign((void**)&data, 4096, paddedSize) != 0) {
            data = (float*)malloc(paddedSize);
        }
        std::fill(data, data + n, 0.0f);
    }

    // Direct access to data
    float& operator[](size_t i) { return data[i]; }
    const float& operator[](size_t i) const { return data[i]; }
    float* begin() { return data; }
    float* end() { return data + size; }

    void swap(SharedBuffer& other) {
        std::swap(data, other.data);
        std::swap(size, other.size);
        std::swap(paddedSize, other.paddedSize);
    }
};

class Fluid
{
public:
    Fluid(float _density, int _numX, int _numY, float _h,
          float _overRelaxation = 1.9, int _numThreads = 4);
    ~Fluid(); // Added destructor to clean up Metal

    // ----------------- start of simulator ------------------------------
    void integrate(float dt, float gravity);
    void integrateMetal(float dt, float gravity);
    void solveIncompressibility(int numIters, float dt);
    void solveIncompressibilityMetal(int numIters, float dt); // Direct GPU version
    void extrapolate();
    void extrapolateMetal();
    void advectVelocityMetal(float dt);
    void advectTracerMetal(float dt);
    void computeVelosityMagnitudeMetal();
    void renderMetal(void* pixels, int width, int height, int mode, bool showTracer, float minVal, float maxVal, float cScale);
    float sampleField(float x, float y, int field);
    float avgU(int i, int j);
    float avgV(int i, int j);
    void simulate(float dt, float gravity, int numIters,
                  const std::function<void(Fluid&)>& correctionStep);
    void computeVelosityMagnitude();
    void advectVelocity(float dt);
    void advectTracer(float dt);
    void applyCorrection(torch::jit::script::Module& model, float inVel);
    void NoCorrection();
    void saveFields(const std::string& filename);
    void updateFluidParameters();
    void syncToTensors();
    void syncToVectors();

    // ----------------- Variables -----------------
    float density;
    int numX;
    int numY;
    int numCells;
    float h;
    float overRelaxation;
    float tt = 0.0;
    int numThreads;
    int num;
    int cnt;

    torch::Device device = torch::kCPU;
    torch::Tensor u_ts, v_ts, p_ts, s_ts, m_ts;
    torch::Tensor newU_ts, newV_ts, newM_ts;
    torch::Tensor net_in_ts; // Persistent input tensor for ML
    torch::Tensor last_cx, last_cy; // Persistent tensors to store last correction
    
    // Step 7: Asynchronous ML Task management
    std::thread m_mlThread;
    std::atomic<bool> m_mlBusy{false};
    std::atomic<bool> m_hasFreshData{false}; 
    int m_stepsRemaining{0}; // New: Spread correction over N frames
    std::mutex m_mlMutex;

    SharedBuffer u;
    SharedBuffer v;
    SharedBuffer newU;
    SharedBuffer newV;
    SharedBuffer Vel;
    SharedBuffer p;
    SharedBuffer s;
    SharedBuffer m;
    SharedBuffer u_corrected;
    SharedBuffer v_corrected;
    SharedBuffer newM;

    // Metal specific
    std::unique_ptr<MetalState> metal;
    bool useMetal = false;
};

#endif // FLUID_H
