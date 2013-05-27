/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "particles.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void particle_mover(particle *d_e, unsigned int *d_e_bm, particle *d_i, unsigned int *d_i_bm, double *Ex, double *Ey) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double me = init_me();     // electron's mass
  static const double mi = init_mi();     // ion's mass
  static const double ds = init_ds();     // spatial step
  static const double dt = init_dt();     // time step
  static const int nnx = init_nnx();      // number of nodes in x dimension
  static const int ncy = init_ncx();      // number of cells in y dimension
  
  unsigned int *h_bm;       // host vector for bookmarks
  int np;                   // number of particles
  
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  
  // device memory
  double *Fx, *Fy;        // force suffered for each particle (electrons or ions)
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for fast_grid_to_particle and leap_frog kernel
  griddim = ncy;
  blockdim = PAR_MOV_BLOCK_DIM;
  
  // define size of shared memory for fast_grid_to_particle kernel
  sh_mem_size = 2*2*nnx*sizeof(double)+2*sizeof(unsigned int);
  
  //---- move electrons
  
  // evaluate number of particles to move (electrons)
  h_bm = (unsigned int *) malloc(2*ncy*sizeof(unsigned int));
  cudaMemcpy(h_bm, d_e_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  np = h_bm[2*ncy-1]-h_bm[0];
  
  // allocate device memory for particle forces (electrons)
  cudaMalloc(&Fx, np*sizeof(double));
  cudaMalloc(&Fy, np*sizeof(double));
  
  // call to fast_grid_to_particle kernel (electrons)
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, -1, ds, d_e, d_e_bm, Ex, Ey, Fx, Fy);
  
  // call to leap_frog_step kernel (electrons)
  leap_frog_step<<<griddim, blockdim>>>(dt, me, d_e, d_e_bm, Fx, Fy);
  
  // free device memory for particle forces (electrons)
  cudaFree(Fx);
  cudaFree(Fy);
  
  //---- move ions  
  
  // evaluate number of particles to move (ions)
  cudaMalloc(&h_bm, 2*ncy*sizeof(unsigned int));
  cudaMemcpy(h_bm, d_i_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  np = h_bm[2*ncy-1]-h_bm[0];
  
  // allocate device memory for particle forces (ions)
  cudaMalloc(&Fx, np*sizeof(double));
  cudaMalloc(&Fy, np*sizeof(double));
  
  // call to fast_grid_to_particle kernel (ions)
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, +1, ds, d_i, d_i_bm, Ex, Ey, Fx, Fy);
  
  // call to fast_grid_to_particle kernel (ions)
  leap_frog_step<<<griddim, blockdim>>>(dt, mi, d_i, d_i_bm, Fx, Fy);
  
  // free device memory for particle forces (ions)
  cudaFree(Fx);
  cudaFree(Fy);
  
  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void fast_grid_to_particle(int nnx, int q, double ds, particle *g_p, unsigned int *g_bm, double *g_Ex, double *g_Ey, double *g_Fx, double *g_Fy) 
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  double *sh_Ex = (double *) sh_mem;                      //
  double *sh_Ey = (double *) &sh_Ex[2*nnx];               // manually set up shared memory variables inside whole shared memory
  unsigned int *sh_bm = (unsigned int *) &sh_Ey[2*nnx];   //
  
  // kernel registers
  particle p;
  double Fx, Fy;
  double distx, disty;
  int ic;
  int jc = blockIdx.x;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load Fields from global memory
  if (blockDim.x >= 2*nnx)
  {
    if (threadIdx.x < 2*nnx)
    {
      sh_Ex[threadIdx.x] = g_Ex[blockIdx.x*nnx+threadIdx.x];
      sh_Ey[threadIdx.x] = g_Ey[blockIdx.x*nnx+threadIdx.x];
    }
    __syncthreads();
  }
  else
  {
    for (int i = threadIdx.x; i < 2*nnx; i+=blockDim.x) 
    {
      sh_Ex[i] = g_Ex[blockIdx.x*nnx+i];
      sh_Ey[i] = g_Ey[blockIdx.x*nnx+i];
    }
    __syncthreads();
  }
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //---- Process batches of particles
  
  for (int i = sh_bm[0]+threadIdx.x; i<=sh_bm[1]; i+=blockDim.x)
  {
    // load particle in registers
    p = g_p[i];
    // calculate x coordinate of the cell that the particle belongs to
    ic = int(p.x/ds);
    // calculate distances from particle to down left vertex of the cell (normalized)
    distx = fabs(double(ic*ds)-p.x)/ds;
    disty = fabs(double(jc*ds)-p.y)/ds;
    // acumulate fields from vertex of the cell to particle position
    if (q == +1)
    {
      // x component of force
      Fx = sh_Ex[ic]*(1.0-distx)*(1.0-disty);
      Fx += sh_Ex[ic+1]*distx*(1.0-disty);
      Fx += sh_Ex[ic+nnx]*(1.0-distx)*disty;
      Fx += sh_Ex[ic+nnx+1]*distx*disty;
      // y component of force
      Fy = sh_Ey[ic]*(1.0-distx)*(1.0-disty);
      Fy += sh_Ey[ic+1]*distx*(1.0-disty);
      Fy += sh_Ey[ic+nnx]*(1.0-distx)*disty;
      Fy += sh_Ey[ic+nnx+1]*distx*disty;
    }
    else if (q == -1)
    {
      // x component of force
      Fx = -sh_Ex[ic]*(1.0-distx)*(1.0-disty);
      Fx -= sh_Ex[ic+1]*distx*(1.0-disty);
      Fx -= sh_Ex[ic+nnx]*(1.0-distx)*disty;
      Fx -= sh_Ex[ic+nnx+1]*distx*disty;
      // y component of force
      Fy = -sh_Ey[ic]*(1.0-distx)*(1.0-disty);
      Fy -= sh_Ey[ic+1]*distx*(1.0-disty);
      Fy -= sh_Ey[ic+nnx]*(1.0-distx)*disty;
      Fy -= sh_Ey[ic+nnx+1]*distx*disty;
    }
    // Store forces in global memory
    g_Fx[i] = Fx;
    g_Fy[i] = Fy;
  }
  
  return;
}

/**********************************************************/

__global__ void leap_frog_step(double dt, double m, particle *g_p, unsigned int *g_bm, double *g_Fx, double *g_Fy) 
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ unsigned int sh_bm[2];   // manually set up shared memory variables inside whole shared memory
  
  // kernel registers
  particle p;
  double Fx, Fy;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //---- Process batches of particles
  
  for (int i = sh_bm[0]+threadIdx.x; i<=sh_bm[1]; i+=blockDim.x)
  {
    // load particle data in registers
    p = g_p[i];
    Fx = g_Fx[i];
    Fy = g_Fy[i];
    
    // move particle
    p.vx += dt*Fx/m;
    p.vy += dt*Fy/m;
    p.x += dt*p.vx;
    p.y += dt*p.vy;
    
    // store particle data in global memory
    g_p[i] = p;
  }
  
  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/

