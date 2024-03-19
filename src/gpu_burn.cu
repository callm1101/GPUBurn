#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <chrono>
#include <iostream>
#include <random>
#include <thread>

// #define SINGLE_PRECISION

#define CUDA_CHECK(err)                                                  \
  do {                                                                   \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)

#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)

#ifdef SINGLE_PRECISION
using real = float;
#else
using real = double;
#endif

constexpr int N = 1 << 10;

struct BurnKernel {
  int burn_seconds;
  int cuda_device;
  std::size_t iterations;
  int numbers;
  double gflops;
  std::atomic<bool> compute_switch;

  thrust::host_vector<real> host_A;
  thrust::host_vector<real> host_B;

  thrust::device_vector<real> device_A;
  thrust::device_vector<real> device_B;
  thrust::device_vector<real> device_C;

  cublasHandle_t handle;

  BurnKernel(int burn_seconds, int cuda_device)
      : burn_seconds(burn_seconds), cuda_device(cuda_device), numbers(0), gflops(0.0), compute_switch(false) {
    CUDA_CHECK(cudaSetDevice(cuda_device));

    this->host_A.resize(N * N);
    this->host_B.resize(N * N);

    std::random_device seed;
    std::mt19937_64 engine(seed());
    std::uniform_real_distribution<> distrib(0.0, 1.0);

    for (int i = 0; i < N * N; ++i) {
      this->host_A[i] = distrib(engine);
      this->host_B[i] = distrib(engine);
    }

    std::size_t free_byte, total_byte;
    CUDA_CHECK(cudaMemGetInfo(&free_byte, &total_byte));

    std::size_t matrix_size = N * N * sizeof(real);
    this->iterations = (0.8 * free_byte - (matrix_size * 2)) / matrix_size;

    this->device_A = this->host_A;
    this->device_B = this->host_B;
    this->device_C.resize(N * N * this->iterations);

    CUBLAS_CHECK(cublasCreate(&this->handle));
  }

  void compute() {
    real alpha = 1.0;
    real beta = 0.0;

    for (std::size_t i = 0; this->compute_switch.load() && i < iterations; i++) {
#ifdef SINGLE_PRECISION
      CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                               thrust::raw_pointer_cast(this->device_A.data()), N,
                               thrust::raw_pointer_cast(this->device_B.data()), N, &beta,
                               thrust::raw_pointer_cast(this->device_C.data() + i * N * N), N));

#else
      CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha,
                               thrust::raw_pointer_cast(this->device_A.data()), N,
                               thrust::raw_pointer_cast(this->device_B.data()), N, &beta,
                               thrust::raw_pointer_cast(this->device_C.data() + i * N * N), N));
#endif
    }

    CUDA_CHECK(cudaDeviceSynchronize());
  }

  void start() {
    this->compute_switch.store(true);
    std::thread compute_thread(&BurnKernel::threadMain, this);
    compute_thread.detach();
  }

  void stop() {
    this->compute_switch.store(false);
    this->gflops /= this->numbers;
  }

  void threadMain() {
    CUDA_CHECK(cudaSetDevice(cuda_device));

    while (this->compute_switch.load()) {
      const auto start = std::chrono::high_resolution_clock::now();

      this->compute();

      const auto end = std::chrono::high_resolution_clock::now();
      const std::chrono::duration<double> elapsed_seconds = end - start;
      this->gflops += (2.0 * N * N * N * this->iterations) / (elapsed_seconds.count() * 1e9);
      this->numbers += 1;
    }
  }

  ~BurnKernel() { cublasDestroy(this->handle); }
};

int main(int argc, char* argv[]) {
  static_cast<void>(argc);
  static_cast<void>(argv);

  int burn_seconds;
  std::cout << "Enter the number of seconds to burn: ";
  std::cin >> burn_seconds;

  int device_count = 0;

  cudaError_t error_id = cudaGetDeviceCount(&device_count);
  if (error_id != cudaSuccess) {
    std::cerr << "Error: couldn't find any CUDA devices\n";
    return EXIT_FAILURE;
  }

  std::cout << "Total no. of GPUs found: " << device_count << std::endl;

  for (int i = 0; i < device_count; ++i) {
    cudaDeviceProp device_prop;
    CUDA_CHECK(cudaGetDeviceProperties(&device_prop, i));
    std::cout << "GPU " << i << ": \"" << device_prop.name << "\"\n";
  }

  std::cout << "Choose the GPU to burn: ";

  int cuda_device;
  std::cin >> cuda_device;

  BurnKernel burn_kernel(burn_seconds, cuda_device);

  burn_kernel.start();

  std::cout << "Burning GPU " << cuda_device << " for " << 0 << " second ...\n";

  for (int i = 0; i < burn_seconds; i++) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
    std::cout << "\e[2K"
              << "\e[1A"
              << "Burning GPU " << cuda_device << " for " << i + 1 << " second ...\n";
  }

  burn_kernel.stop();

  std::cout << "GPU " << cuda_device << " burned at " << burn_kernel.gflops << " GFLOPS\n";

  return 0;
}
