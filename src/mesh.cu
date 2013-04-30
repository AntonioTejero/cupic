/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "cc.h"

/************************ FUNCTION DEFINITIONS ***********************/

 fast_particle_to_grid(int ncx, int ncy, double dx, double dy, double *rho, particle *e, unsigned int *e_bm, particle *i, unsigned int *i_bm) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  charge_deposition(ncx, ncy, dx, dy, rho, e, e_bm, i, i_bm);
  
  return;
}

/**********************************************************/

__global__ void charge_deposition(int ncx, int ncy, double dx, double dy, double *rho, particle *e, unsigned int *e_bm, particle *i, unsigned int *i_bm)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ double sh_partial_rho[2*ncx]
  __shared__ unsigned int sh_e_bm[2], sh_i_bm[2];
  
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
      partial_rho[threadIdx.x] = 0.0;
    }
  } else
  {
    for (int i = threadIdx.x; i < 2*ncx; i+=blockDim.x)
    {
      partial_rho[i] = 0.0;
    }
  }
  __synctrheads();
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_e_bm[threadIdx.x] = e_bm[blockIdx.x*2+threadIdx.x];
    sh_i_bm[threadIdx.x] = i_bm[blockIdx.x*2+threadIdx.x];
  }
  __synctrheads();
  
  //--- deposition of charge
  
  // electron deposition
  
  for (int i = sh_e_bm[0]+threadIdx.x; i<=sh_e_bm[1]; i+=blockDim.x)
  {
    // load electron in registers
    p = e[i];
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
    p = i[i];
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
  __synctrheads();
  
  // ccc
  
  if (threadIdx.x < 2)
  {
    sh_partial_rho[threadIdx.x*ncx] += sh_partial_rho[(threadIdx.x+1)*ncx-1];
    sh_partial_rho[(threadIdx.x+1)*ncx-1] = sh_partial_rho[threadIdx.x*ncx];
  }
  __synctrheads();
  
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