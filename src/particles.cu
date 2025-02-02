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

void particle_mover(particle *d_e, int *d_e_bm, particle *d_i, int *d_i_bm, double *Ex, double *Ey) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double me = init_me();     // electron's mass
  static const double mi = init_mi();     // ion's mass
  static const double qe = init_qe();     // electron's charge
  static const double qi = init_qi();     // ions's charge
  static const double ds = init_ds();     // spatial step
  static const double dt = init_dt();     // time step
  static const int nnx = init_nnx();      // number of nodes in x dimension
  static const int ncy = init_ncy();      // number of cells in y dimension
  
  int np;                                 // number of particles
  
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  cudaError_t cuError;
  
  // device memory
  double *Fx, *Fy;        // force suffered for each particle (electrons or ions)
  
  /*----------------------------- function body -------------------------*/
  
  // set dimensions of grid of blocks and blocks of threads for fast_grid_to_particle and leap_frog kernel
  griddim = ncy;
  blockdim = PAR_MOV_BLOCK_DIM;
  
  // define size of shared memory for fast_grid_to_particle kernel
  sh_mem_size = 2*2*nnx*sizeof(double)+2*sizeof(int);
  
  //---- move electrons
  
  // evaluate number of particles to move (electrons)
  np = number_of_particles(d_e_bm);
  
  // allocate device memory for particle forces (electrons)
  cuError = cudaMalloc((void **) &Fx, np*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc((void **) &Fy, np*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // call to fast_grid_to_particle kernel (electrons)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, qe, ds, d_e, d_e_bm, Ex, Ey, Fx, Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // call to leap_frog_step kernel (electrons)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim>>>(dt, me, d_e, d_e_bm, Fx, Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // free device memory for particle forces (electrons)
  cuError = cudaFree(Fx);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(Fy);
  cu_check(cuError, __FILE__, __LINE__);
  
  //---- move ions  
  
  // evaluate number of particles to move (ions)
  np = number_of_particles(d_i_bm);
  
  // allocate device memory for particle forces (ions)
  cuError = cudaMalloc((void **) &Fx, np*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc((void **) &Fy, np*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // call to fast_grid_to_particle kernel (ions)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, qi, ds, d_i, d_i_bm, Ex, Ey, Fx, Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // call to leap_frog_step kernel (ions)
  cudaGetLastError();
  leap_frog_step<<<griddim, blockdim>>>(dt, mi, d_i, d_i_bm, Fx, Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // free device memory for particle forces (ions)
  cuError = cudaFree(Fx);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(Fy);
  cu_check(cuError, __FILE__, __LINE__);
  
  return;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void fast_grid_to_particle(int nnx, int q, double ds, particle *g_p, int *g_bm, double *g_Ex, double *g_Ey, double *g_Fx, double *g_Fy) 
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  
  double *sh_Ex = (double *) sh_mem;          //
  double *sh_Ey = (double *) &sh_Ex[2*nnx];   // manually set up shared memory variables inside whole shared memory
  int *sh_bm = (int *) &sh_Ey[2*nnx];         //
  
  // kernel registers
  particle p;
  double Fx, Fy;
  double distx, disty;
  int ic;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load Fields from global memory
  for (int i = threadIdx.x; i < 2*nnx; i += blockDim.x) {
    sh_Ex[i] = g_Ex[blockIdx.x*nnx+i];
    sh_Ey[i] = g_Ey[blockIdx.x*nnx+i];
  }
  __syncthreads();
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2) {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //---- Process batches of particles
  if (sh_bm[0] >= 0 && sh_bm[1] >= 0) {
    for (int i = sh_bm[0]+threadIdx.x; i <= sh_bm[1]; i += blockDim.x) {
      // load particle in registers
      p = g_p[i];
      // calculate x coordinate of the cell that the particle belongs to
      ic = __double2int_rd(p.x/ds);
      // calculate distances from particle to down left vertex of the cell (normalized)
      distx = fabs(double(ic*ds)-p.x)/ds;
      disty = fabs(double(blockDim.x*ds)-p.y)/ds;
      // acumulate fields from vertex of the cell to particle position
      if (q == +1) {
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
      else if (q == -1) {
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
  }
  
  return;
}

/**********************************************************/

__global__ void leap_frog_step(double dt, double m, particle *g_p, int *g_bm, double *g_Fx, double *g_Fy) 
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_bm[2];   // manually set up shared memory variables inside whole shared memory
  
  // kernel registers
  particle p;
  double Fx, Fy;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory variables
  
  // load bin bookmarks from global memory
  if (threadIdx.x < 2) {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //---- Process batches of particles

  if (sh_bm[0] >= 0 && sh_bm[1] >= 0) {
    for (int i = sh_bm[0]+threadIdx.x; i <= sh_bm[1]; i += blockDim.x) {
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
  }
  
  return;
}

/**********************************************************/



/******************** DEVICE FUNCTION DEFINITIONS ********************/

