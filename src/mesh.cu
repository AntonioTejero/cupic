/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "mesh.h"

/************************ FUNCTION DEFINITIONS ***********************/

void fast_particle_to_grid(int ncx, int ncy, double dx, double dy, double *rho, particle *elec, unsigned int *e_bm, particle *ions, unsigned int *i_bm) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for particle defragmentation kernel
  griddim = ncy;
  blockdim = CHARGE_DEP_BLOCK_DIM;
  
  // define size of shared memory for charge_deposition kernel
  sh_mem_size = 2*ncx*sizeof(double)+4*sizeof(unsigned int);
  
  charge_deposition<<<griddim, blockdim, sh_mem_size>>>(ncx, ncy, dx, dy, rho, elec, e_bm, ions, i_bm);
  
  return;
}

/**********************************************************/

__global__ void charge_deposition(int ncx, int ncy, double dx, double dy, double *rho, particle *elec, unsigned int *e_bm, particle *ions, unsigned int *i_bm)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  double *sh_partial_rho = (double *) sh_mem;                       //
  unsigned int *sh_e_bm = (unsigned int *) &sh_partial_rho[2*ncx];  // manually set up shared memory variables inside whole shared memory
  unsigned int *sh_i_bm = (unsigned int *) &sh_e_bm[2];             //
  
  // kernel registers
  particle p;
  double distx, disty;
  int ic;
  int jc = blockIdx.x;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // initialize charge density in shared memory to 0.0
  if (blockDim.x >= 2*ncx)
  {
    if (threadIdx.x < 2*ncx)
    {
      sh_partial_rho[threadIdx.x] = 0.0;
    }
  } else
  {
    for (int i = threadIdx.x; i < 2*ncx; i+=blockDim.x)
    {
      sh_partial_rho[i] = 0.0;
    }
  }
  __syncthreads();
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_e_bm[threadIdx.x] = e_bm[blockIdx.x*2+threadIdx.x];
    sh_i_bm[threadIdx.x] = i_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //--- deposition of charge
  
  // electron deposition
  
  for (int i = sh_e_bm[0]+threadIdx.x; i<=sh_e_bm[1]; i+=blockDim.x)
  {
    // load electron in registers
    p = elec[i];
    // calculate x coordinate of the cell that the electron belongs to
    ic = int(p.x/dx);
    // calculate distances from particle to down left vertex of the cell
    distx = fabs(double(ic*dx)-p.x);
    disty = fabs(double(jc*dy)-p.y);
    // acumulate charge in partial rho
    atomicSub(sh_partial_rho+ic, (1.0-distx/dx)*(1.0-disty/dy));    //down left vertex
    atomicSub(sh_partial_rho+ic+1, (distx/dx)*(1.0-disty/dy));      //down right vertex
    atomicSub(sh_partial_rho+ic+ncx, (1.0-distx/dx)*(disty/dy));    //top left vertex
    atomicSub(sh_partial_rho+ic+ncx+1, (distx/dx)*(disty/dy));      //top right vertex
  }
  
  // ion deposition
  
  for (int i = sh_i_bm[0]+threadIdx.x; i<=sh_i_bm[1]; i+=blockDim.x)
  {
    // load electron in registers
    p = ions[i];
    // calculate x coordinate of the cell that the electron belongs to
    ic = int(p.x/dx);
    // calculate distances from particle to down left vertex of the cell
    distx = fabs(double(ic*dx)-p.x);
    disty = fabs(double(jc*dy)-p.y);
    // acumulate charge in partial rho
    atomicAdd(sh_partial_rho+ic, (1.0-distx/dx)*(1.0-disty/dy));    //down left vertex
    atomicAdd(sh_partial_rho+ic+1, (distx/dx)*(1.0-disty/dy));      //down right vertex
    atomicAdd(sh_partial_rho+ic+ncx, (1.0-distx/dx)*(disty/dy));    //top left vertex
    atomicAdd(sh_partial_rho+ic+ncx+1, (distx/dx)*(disty/dy));      //top right vertex
  }
  __syncthreads();
  
  // ccc
  
  if (threadIdx.x < 2)
  {
    sh_partial_rho[threadIdx.x*ncx] += sh_partial_rho[(threadIdx.x+1)*ncx-1];
    sh_partial_rho[(threadIdx.x+1)*ncx-1] = sh_partial_rho[threadIdx.x*ncx];
  }
  __syncthreads();
  
  //---- acumulation of charge
  
  if (blockDim.x >= 2*ncx)
  {
    if (threadIdx.x < 2*ncx)
    {
      atomicAdd(rho+blockIdx.x*ncx+threadIdx.x, sh_partial_rho[threadIdx.x]);
    }
  } else
  {
    for (int i = threadIdx.x; i < 2*ncx; i+=blockDim.x)
    {
      atomicAdd(rho+blockIdx.x*ncx+i, sh_partial_rho[i]);
    }
  }
  
  
  return;
}

/**********************************************************/

__device__ double atomicAdd(double* address, double val)
{
  /*--------------------------- function variables -----------------------*/
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  
  /*----------------------------- function body -------------------------*/
  do 
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  
  return __longlong_as_double(old);
}

/**********************************************************/

__device__ double atomicSub(double* address, double val)
{
  /*--------------------------- function variables -----------------------*/
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;
  
  /*----------------------------- function body -------------------------*/
  do 
  {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val - __longlong_as_double(assumed)));
  } while (assumed != old);
  
  return __longlong_as_double(old);
}

/**********************************************************/

void poisson_solver(int ncx, int ncy, double ds, double max_error, double epsilon0, double *rho, double *phi) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  dim3 blockdim, griddim;
  double *h_block_error;
  double error = max_error*10;
  size_t sh_mem_size;
  int count = max(ncx, ncy);
  
  // device memory
  double *d_block_error;
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for jacobi kernel
  blockdim.x = ncx;
  blockdim.y = 1024/ncx;
  griddim = (ncy-2)/blockdim.y;
  
  // define size of shared memory for jacobi_iteration kernel
  sh_mem_size = (2*blockdim.x*(blockdim.y+1)+blockdim.y)*sizeof(double);
  
  // allocate host memory
  h_block_error = new double[griddim.x];
  
  // allocate device memory
  cudaMalloc(&d_block_error, griddim.x*sizeof(double));
  
  // execute jacobi iterations until solved
  while(count<=0 && error>=max_error)
  {
    // launch kernel for performing one jacobi iteration
    jacobi_iteration<<<griddim, blockdim, sh_mem_size>>>(blockdim, ds, epsilon0, rho, phi, d_block_error);
    
    // copy device memory to host memory for analize errors
    cudaMemcpy(h_block_error, d_block_error, griddim.x*sizeof(double), cudaMemcpyDeviceToHost);
    
    // evaluate max error in the iteration
    error = 0;
    for (int i = 0; i < griddim.x; i++)
    {
      if (h_block_error[i]>error) error = h_block_error[i];
    }
    
    // actualize counter
    count--;
  }
  
  return;
}

/**********************************************************/

__global__ void jacobi_iteration (dim3 blockdim, double ds, double epsilon0, double *rho, double *phi, double *block_error)
{
  // shared memory
  double *phi_old = (double *) sh_mem;                              //
  double *error = (double *) &phi_old[blockdim.x*(blockdim.y+2)];   // manually set up shared memory variables inside whole shared memory
  double *aux_shared = (double *) &error[blockdim.x*blockdim.y];    //
  
//   __shared__ double phi_old[BLOCKDIMX*(BLOCKDIMY+2)];
//   __shared__ double aux_shared[BLOCKDIMY];
//   __shared__ double error[BLOCKDIMX*BLOCKDIMY];
  
  // registers
  double phi_new, rho_dummy;
  int global_mem_index = blockDim.x + blockIdx.x*(blockDim.x*blockDim.y) + threadIdx.y*blockDim.x + threadIdx.x;
  int shared_mem_index = blockDim.x + threadIdx.y*blockDim.x + threadIdx.x;
  int thread_index = threadIdx.x + threadIdx.y*blockDim.x;
  
  // kernel body
  
  // load phi data from global memory to shared memory
  phi_old[shared_mem_index] = phi[global_mem_index];
  
  // load comunication zones into shared memory
  if (threadIdx.y == 0)
  {
    phi_old[shared_mem_index-blockDim.x] = phi[global_mem_index-blockDim.x];
  }
  if (threadIdx.y == blockDim.y-1)
  {
    phi_old[shared_mem_index+blockDim.x] = phi[global_mem_index+blockDim.x];
  }
  // load charge density data into registers
  rho_dummy = ds*ds*rho[global_mem_index]/epsilon0;
  __syncthreads();
  
  // actualize cyclic contour conditions
  if (threadIdx.x == 0)
  {
    phi_new = 0.25*(rho_dummy + phi_old[shared_mem_index+blockDim.x-2] + phi_old[shared_mem_index+1] + phi_old[shared_mem_index+blockDim.x]+phi_old[shared_mem_index-blockDim.x]);
    aux_shared[threadIdx.y] = phi_new;
  }
  __syncthreads();
  if (threadIdx.x == blockDim.x-1)
  {
    phi_new = aux_shared[threadIdx.y];
  }
  
  // actualize interior mesh points
  if (threadIdx.x != 0 && threadIdx.x != blockDim.x-1)
  {
    phi_new = 0.25*(rho_dummy + phi_old[shared_mem_index-1] + phi_old[shared_mem_index+1] + phi_old[shared_mem_index+blockDim.x]+phi_old[shared_mem_index-blockDim.x]);
  }
  __syncthreads();
  
  // evaluate local errors
  error[thread_index] = fabs(phi_new-phi_old[shared_mem_index]);
  __syncthreads();
  
  // reduction for obtaining maximum error in current block
  for (int stride = 1; stride < blockDim.x*blockDim.y; stride <<= 1)
  {
    if (thread_index%(stride*2) == 0)
    {
      if (thread_index+stride<blockDim.x*blockDim.y)
      {
        if (error[thread_index]<error[thread_index+stride]) error[thread_index] = error[thread_index+stride];
      }
    }
    __syncthreads();
  }
  
  // store block error in global memory
  if (thread_index == 0)
  {
    block_error[blockIdx.x] = error[0];
  }
  
  // store new values of phi in global memory
  phi[global_mem_index] = phi_new;
  
  return;
}

/**********************************************************/
