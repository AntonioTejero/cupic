/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "cc.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void cc (double t, int *d_e_bm, particle **d_e, int *d_i_bm, particle **d_i, double *d_Ex, double *d_Ey)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double me = init_me();                     //
  static const double mi = init_mi();                     // particle
  static const double kti = init_kti();                   // properties
  static const double kte = init_kte();                   //
  
  static const double dt = init_dt();                     // time step
  static const double dtin_e = init_dtin_e();             // time between electron insertions sqrt(2.0*PI*m_e/kT_e)/(n*Lx*dz)
  static const double dtin_i = init_dtin_i();             // time between ion insertions sqrt(2.0*PI*m_i/kT_i)/(n*Lx*dz)
  
  static const double Lx = init_Lx();                     //
  static const double Ly = init_Ly();                     // geometric properties of simulation
  static const double ds = init_ds();                     //
  
  static const int nnx = init_nnx();                      // number of nodes in x dimension
  static const int nny = init_nny();                      // number of nodes in y dimension
  static const int ncy = init_ncy();                      // number of cells in y dimension
  
  static double tin_e = dtin_e;                           // time for next electron insertion
  static double tin_i = dtin_i;                           // time for next ion insertion
  double fpt = t+dt;                                      // future position time
  double fvt = t+0.5*dt;                                  // future velocity time
  int in_e = 0;                                           // number of electron added at plasma frontier
  int in_i = 0;                                           // number of ions added at plasma frontier
  int out_e_l, out_e_r, out_i_l, out_i_r;                 // number of electrons and ions withdrawn at probe (l) and at plasma frontier (r)
  
  int h_e_bm[2*ncy], h_i_bm[2*ncy];                       // old particle bookmarks
  int h_e_new_bm[2*ncy], h_i_new_bm[2*ncy];               // new particle bookmarks
  int length;                                             // length of particle vectors
  
  double *h_Ex, *h_Ey;                                    // host memory for electric fields
  double Epx, Epy;                                        // fields at particle position
  int ic, jc;                                             // indices of particle cell
  double distx, disty;                                    // distance from particle to nodes  
  particle *dummy_p;                                      // dummy vector for particle storage
  
  static gsl_rng * rng = gsl_rng_alloc(gsl_rng_default);  //default random number generator (gsl)
  
  cudaError cuError;
  
  // device memory
  int *d_e_new_bm, *d_i_new_bm;      // new particle bookmarks (have to be allocated in device memory)

  /*----------------------------- function body -------------------------*/

  //---- sorting and cyclic contour conditions
  
  // allocate device memory for new particle bookmarks
  cuError = cudaMalloc (&d_e_new_bm, 2*ncy*sizeof(int));
  cu_check(cuError);
  cuError = cudaMalloc (&d_i_new_bm, 2*ncy*sizeof(int));  
  cu_check(cuError);
  
  // sort particles with bining algorithm, also apply cyclic contour conditions during particle defragmentation
  particle_bining(Lx, ds, ncy, d_e_bm, d_e_new_bm, *d_e);
  particle_bining(Lx, ds, ncy, d_i_bm, d_i_new_bm, *d_i);
  
  // copy new and old bookmark to host memory
  cuError = cudaMemcpy (h_e_bm, d_e_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError);
  cuError = cudaMemcpy (h_i_bm, d_i_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError);
  cuError = cudaMemcpy (h_e_new_bm, d_e_new_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError);
  cuError = cudaMemcpy (h_i_new_bm, d_i_new_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError);
  
  //---- absorbent/emitter contour conditions
  
  // calculate number of electrons and ions that flow into the simulation
  if(tin_e < fpt) in_e = 1 + int((fpt-tin_e)/dtin_e);
  if(tin_i < fpt) in_i = 1 + int((fpt-tin_i)/dtin_i);
  
  // calculate number of electrons and ions that flow out of the simulation
  out_e_l = h_e_new_bm[0]-h_e_bm[0];
  out_e_r = h_e_bm[2*ncy-1]-h_e_new_bm[2*ncy-1];
  out_i_l = h_i_new_bm[0]-h_i_bm[0];
  out_i_r = h_i_bm[2*ncy-1]-h_i_new_bm[2*ncy-1];
  
  //-- electrons
  if (out_e_l != 0 || out_e_r != 0 || in_e != 0)
  {
    // move particles to host dummy vector
    length = h_e_new_bm[2*ncy-1]-h_e_new_bm[0]+1;
    dummy_p = (particle*) malloc((length+in_e)*sizeof(particle));
    cuError = cudaMemcpy(dummy_p, *d_e+h_e_new_bm[0], length*sizeof(particle), cudaMemcpyDeviceToHost);
    cu_check(cuError);
    cuError = cudaFree(*d_e);
    cu_check(cuError);
    
    // actualize bookmarks (left removed particles)
    if (out_e_l != 0)
    {
      for (int k = 0; k < 2*ncy; k++) 
      {
        h_e_new_bm[k] -= out_e_l;
      }
    }
    
    // add particles
    if (in_e != 0)
    {
      // actualize length of particle vector
      length += in_e;
      
      // copy fields data from device to host for simple push
      h_Ex = (double *) malloc(nnx*nny*sizeof(double));
      h_Ey = (double *) malloc(nnx*nny*sizeof(double));
      cuError = cudaMemcpy (h_Ex, d_Ex, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cu_check(cuError);
      cuError = cudaMemcpy (h_Ey, d_Ey, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cu_check(cuError);
      
      // create new particles
      for (int k = h_e_new_bm[2*ncy-1]+1; k < length; k++) 
      {
        //initialize particles
        dummy_p[k].x = gsl_rng_uniform_pos(rng)*Lx;
        dummy_p[k].y = Ly;         
        dummy_p[k].vx = gsl_ran_gaussian(rng, sqrt(kte/me));
        dummy_p[k].vy = gsl_ran_rayleigh(rng, sqrt(kte/me));
        
        // calculate cell index
        ic = int (dummy_p[k].x/ds);
        jc = ncy-1;
        
        // calculate distances from particle to down left vertex (normalized to ds)
        distx = fabs(double(ic)*ds-dummy_p[k].x)/ds;
        disty = 1.0;
        
        // interpolate fields from nodes to particle
        Epx = h_Ex[ic+jc*nnx]*(1.0-distx)*(1.0-disty);
        Epx += h_Ex[ic+1+jc*nnx]*distx*(1.0-disty);
        Epx += h_Ex[ic+(jc+1)*nnx]*(1.0-distx)*disty;
        Epx += h_Ex[ic+1+(jc+1)*nnx]*distx*disty;
        
        Epy = h_Ey[ic+jc*nnx]*(1.0-distx)*(1.0-disty);
        Epy += h_Ey[ic+1+jc*nnx]*distx*(1.0-disty);
        Epy += h_Ey[ic+(jc+1)*nnx]*(1.0-distx)*disty;
        Epy += h_Ey[ic+1+(jc+1)*nnx]*distx*disty;
        
        // simple push
        dummy_p[k].x += (fpt-tin_e)*dummy_p[k].vx;
        dummy_p[k].y += (fpt-tin_e)*dummy_p[k].vy;  // comprobar que no atraviesa una celda
        dummy_p[k].vx -= (fvt-tin_e)*Epx/me;
        dummy_p[k].vy -= (fvt-tin_e)*Epy/me;
        
        // actualize time for next particle insertion
        tin_e += dtin_e;
      }
      
      // free host memory for fields data
      free(h_Ex);
      free(h_Ey);
      
      // move end bookmark of last bin_bookmark (added particles)
      h_e_new_bm[2*ncy-1] += in_e;
    }
    
    // copy new particles to device memory
    cuError = cudaMalloc(d_e, length*sizeof(particle));
    cu_check(cuError);
    cuError = cudaMemcpy(*d_e, dummy_p, length*sizeof(particle), cudaMemcpyHostToDevice);
    cu_check(cuError);
    free(dummy_p);
  }
  
  //-- ions
  if (out_i_l != 0 || out_i_r != 0 || in_i != 0)
  {
    // move particles to host dummy vector
    length = h_i_new_bm[2*ncy-1]-h_i_new_bm[0]+1;
    dummy_p = (particle*) malloc((length+in_i)*sizeof(particle));
    cuError = cudaMemcpy(dummy_p, *d_i+h_i_new_bm[0], length*sizeof(particle), cudaMemcpyDeviceToHost);
    cu_check(cuError);
    cuError = cudaFree(*d_i);
    cu_check(cuError);
    
    // actualize bookmarks (left removed particles)
    if (out_i_l != 0)
    {
      for (int k = 0; k < 2*ncy; k++) 
      {
        h_i_new_bm[k] -= out_i_l;
      }
    }
    
    // add particles
    if (in_i != 0)
    {
      // actualize length of particle vector
      length += in_i;
      
      // copy fields data from device to host for simple push
      h_Ex = (double *) malloc(nnx*nny*sizeof(double));
      h_Ey = (double *) malloc(nnx*nny*sizeof(double));
      cuError = cudaMemcpy (h_Ex, d_Ex, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cu_check(cuError);
      cuError = cudaMemcpy (h_Ey, d_Ey, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cu_check(cuError);
      
      // create new particles
      for (int k = h_i_new_bm[2*ncy-1]+1; k < length; k++) 
      {
        //initialize particles
        dummy_p[k].x = gsl_rng_uniform_pos(rng)*Lx;
        dummy_p[k].y = Ly;
        dummy_p[k].vx = gsl_ran_gaussian(rng, sqrt(kti/mi));
        dummy_p[k].vy = gsl_ran_rayleigh(rng, sqrt(kti/mi));
        
        // calculate cell index
        ic = int (dummy_p[k].x/ds);
        jc = ncy-1;
        
        // calculate distances from particle to down left vertex (normalized to ds)
        distx = fabs(double(ic)*ds-dummy_p[k].x)/ds;
        disty = 1.0;
        
        // interpolate fields from nodes to particle
        Epx = h_Ex[ic+jc*nnx]*(1.0-distx)*(1.0-disty);
        Epx += h_Ex[ic+1+jc*nnx]*distx*(1.0-disty);
        Epx += h_Ex[ic+(jc+1)*nnx]*(1.0-distx)*disty;
        Epx += h_Ex[ic+1+(jc+1)*nnx]*distx*disty;
        
        Epy = h_Ey[ic+jc*nnx]*(1.0-distx)*(1.0-disty);
        Epy += h_Ey[ic+1+jc*nnx]*distx*(1.0-disty);
        Epy += h_Ey[ic+(jc+1)*nnx]*(1.0-distx)*disty;
        Epy += h_Ey[ic+1+(jc+1)*nnx]*distx*disty;
        
        // simple push
        dummy_p[k].x += (fpt-tin_i)*dummy_p[k].vx;
        dummy_p[k].y += (fpt-tin_i)*dummy_p[k].vy; // comprobar que no atraviesa una celda
        dummy_p[k].vx += (fvt-tin_i)*Epx/mi;
        dummy_p[k].vy += (fvt-tin_i)*Epy/mi;
        
        // actualize time for next particle insertion
        tin_i += dtin_i;
      }
      
      // free host memory for fields data
      free(h_Ex);
      free(h_Ey);
      
      // move end bookmark of last bin_bookmark (added particles)
      h_i_new_bm[2*ncy-1] += in_i;
    }
    
    // copy new particles to device memory
    cuError = cudaMalloc(d_i, length*sizeof(particle));
    cu_check(cuError);
    cuError = cudaMemcpy(*d_i, dummy_p, length*sizeof(particle), cudaMemcpyHostToDevice);
    cu_check(cuError);
    free(dummy_p);
  }
  
  // copy new bookmarks to device memory
  cuError = cudaMemcpy (d_e_bm, h_e_new_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError);
  cuError = cudaMemcpy (d_i_bm, h_i_new_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError);
  
  return;
}

/**********************************************************/

void particle_bining(double Lx, double ds, int ncy, int *bm, int *new_bm, particle *p)
{
  /*--------------------------- function variables -----------------------*/

  dim3 griddim, blockdim;

  /*----------------------------- function body --------------------------*/

  // set dimensions of grid of blocks and blocks of threads for particle defragmentation kernel
  griddim = ncy;
  blockdim = BINING_BLOCK_DIM;
  
  // execute kernel for defragmentation of particles
  cudaGetLastError();
  pDefragDown<<<griddim, blockdim>>>(ds, bm, new_bm, p);
  cu_sync_check();
  cudaGetLastError();
  pDefragUp<<<griddim, blockdim>>>(ds, bm, new_bm, p);
  cu_sync_check();
  
  // set dimension of grid of blocks for particle rebracketing kernel
  griddim = ncy-1;
  
  // execute kernel for rebracketing of particles
  cudaGetLastError();
  particle_rebracketing<<<griddim, blockdim>>>(bm, new_bm, p);
  cu_sync_check();
  
  return;
}

/**********************************************************/


/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void pDefragDown(double ds, int *g_bm, int *g_new_bm, particle *g_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ particle sh_p[BINING_BLOCK_DIM];
  __shared__ int sh_bm[2];
  __shared__ int tail, i, i_shifted;
  
  // kernel registers
  int swap_index;
  particle reg_p, tmp_p;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  
  // load bin bookmarks
  if (threadIdx.x < 2) {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }

  // initialize batch parameters
  if (0 == threadIdx.x) {
    tail = 0;
    i = sh_bm[0];
    i_shifted = i + blockDim.x;
  }
  __syncthreads();

  //---- cleanup first batch of "-" particles

  // load shared memory batch and register batch
  sh_p[threadIdx.x] = g_p[i+threadIdx.x];
  __syncthreads();
  reg_p = g_p[i_shifted+threadIdx.x];

  // analize registers particles and swap
  if (__double2int_rd(reg_p.y/ds) < (int) blockIdx.x) {
    do {
      swap_index = atomicAdd(&tail, 1);
    } while (__double2int_rd(sh_p[swap_index].y/ds) < (int) blockIdx.x);
    tmp_p = reg_p;
    reg_p = sh_p[swap_index];
    sh_p[swap_index] = tmp_p;
  }
  __syncthreads();

  // store results in global memory
  g_p[i+threadIdx.x] = sh_p[threadIdx.x];
  __syncthreads();
  g_p[i_shifted+threadIdx.x] = reg_p;
  __syncthreads();

  // reset tail parameter
  if (0 == threadIdx.x) {
    tail = 0;
  }

  //---- start "-" defrag

  while (i_shifted <= sh_bm[1]) {
    // load shared batch
    sh_p[threadIdx.x] = g_p[i+threadIdx.x];
    __syncthreads();

    // load, analize and swap register batch
    if (i_shifted+threadIdx.x <= sh_bm[1]) {
      reg_p = g_p[i_shifted+threadIdx.x];
      if (__double2int_rd(reg_p.y/ds) < (int) blockIdx.x) {
        swap_index = atomicAdd(&tail, 1);
        tmp_p = reg_p;
        reg_p = sh_p[swap_index];
        sh_p[swap_index] = tmp_p;
      }
    }
    __syncthreads();

    // store results in global memory
    g_p[i+threadIdx.x] = sh_p[threadIdx.x];
    __syncthreads();

    if (i_shifted+threadIdx.x <= sh_bm[1]) {
      g_p[i_shifted+threadIdx.x] = reg_p;
    }
    __syncthreads();

    // actualize batch parameters
    if (0 == threadIdx.x) {
      i += tail;
      i_shifted += blockDim.x;
      tail = 0;
    }
  }

  //---- store new left bookmarks in global memory

  if (0 == threadIdx.x) {
    g_new_bm[blockIdx.x*2] = i;
  }

  return;
}

/**********************************************************/

__global__ void pDefragUp(double ds, int *g_bm, int *g_new_bm, particle *g_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ particle sh_p[BINING_BLOCK_DIM];
  __shared__ int sh_bm[2];
  __shared__ int tail, i, i_shifted;
  
  // kernel registers
  int swap_index;
  particle reg_p, tmp_p;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  
  // load bin bookmarks
  if (int(threadIdx.x) < 2) {
    sh_bm[int(threadIdx.x)] = g_bm[int(blockIdx.x)*2+int(threadIdx.x)];
  }
  
  // initialize batch parameters
  if (0 == int(threadIdx.x)) {
    tail = 0;
    i = sh_bm[1];
    i_shifted = i - blockDim.x;
  }
  __syncthreads();
  
  //---- cleanup last batch of "+" particles
  
  // load shared memory batch and register batch
  sh_p[int(threadIdx.x)] = g_p[i-int(threadIdx.x)];
  __syncthreads();
  reg_p = g_p[i_shifted-int(threadIdx.x)];
  
  // analize registers particles and swap
  if (__double2int_rd(reg_p.y/ds) > (int) blockIdx.x) {
    do {
      swap_index = atomicAdd(&tail, 1);
    } while (__double2int_rd(sh_p[swap_index].y/ds) > (int) blockIdx.x);
    tmp_p = reg_p;
    reg_p = sh_p[swap_index];
    sh_p[swap_index] = tmp_p;
  }
  __syncthreads();
  
  // store results in global memory
  g_p[i-int(threadIdx.x)] = sh_p[int(threadIdx.x)];
  __syncthreads();
  g_p[i_shifted-int(threadIdx.x)] = reg_p;
  __syncthreads();
  
  // reset tail parameter
  if (0 == int(threadIdx.x)) {
    tail = 0;
  }
  
  //---- start "+" defrag
  
  while (i_shifted >= sh_bm[0]) {
    // load shared batch
    sh_p[int(threadIdx.x)] = g_p[i-int(threadIdx.x)];
    __syncthreads();
    
    // load, analize and swap register batch
    if (i_shifted-int(threadIdx.x) >= sh_bm[0]) {
      reg_p = g_p[i_shifted-int(threadIdx.x)];
      if (__double2int_rd(reg_p.y/ds) > (int) blockIdx.x) {
        swap_index = atomicAdd(&tail, 1);
        tmp_p = reg_p;
        reg_p = sh_p[swap_index];
        sh_p[swap_index] = tmp_p;
      }
    }
    __syncthreads();
    
    // store results in global memory
    g_p[i-int(threadIdx.x)] = sh_p[int(threadIdx.x)];
    __syncthreads();
    
    if (i_shifted-int(threadIdx.x) >= sh_bm[0]) {
      g_p[i_shifted-int(threadIdx.x)] = reg_p;
    }
    __syncthreads();
    
    // actualize batch parameters
    if (0 == int(threadIdx.x)) {
      i -= tail;
      i_shifted -= blockDim.x;
      tail = 0;
    }
  }
  
  //---- store new right bookmarks in global memory
  
  if (0 == int(threadIdx.x)) {
    g_new_bm[int(blockIdx.x)*2+1] = i;
  }
  
  return;
}

/**********************************************************/

__global__ void particle_rebracketing(int *bm, int *new_bm, particle *p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_old_bm[2];        // bookmarks before defragmentation (also used to store bookmarks after rebracketing) (bin_end, bin_start)
  __shared__ int sh_new_bm[2];        // bookmarks after particle defragmentation (bin_end, bin_start)
  __shared__ int nswaps;              // number of swaps each bin frontier needs
  __shared__ int tpb;                 // threads per block
  // kernel registers
  particle p_dummy;                   // dummy particle for swapping
  int stride = 1+threadIdx.x;         // offset stride for each thread to swap the correct particle
  
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  
  // load old and new bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_old_bm[threadIdx.x] = bm[1+blockIdx.x*2+threadIdx.x];
    sh_new_bm[threadIdx.x] = new_bm[1+blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  // set tpb variable and evaluate number of swaps needed for each bin frontier
  if (threadIdx.x == 0)
  {
    tpb = blockDim.x;
    nswaps = ( (sh_old_bm[0]-sh_new_bm[0])<(sh_new_bm[1]-sh_old_bm[1]) ) ? (sh_old_bm[0]-sh_new_bm[0]) : (sh_new_bm[1]-sh_old_bm[1]);
  }
  __syncthreads();
  
  //---- if number of swaps needed is greater than the number of threads per block:
  
  while (nswaps >= tpb)
  {
    // swapping of tpb particles
    p_dummy = p[sh_new_bm[0]+stride];
    p[sh_new_bm[0]+stride] = p[sh_new_bm[1]-stride];
    p[sh_new_bm[1]-stride] = p_dummy;
    __syncthreads();
    
    // actualize shared new bookmarks
    if (threadIdx.x == 0)
    {
      sh_new_bm[0] += tpb;
      sh_new_bm[1] -= tpb;
      nswaps -= tpb;
    }
    __syncthreads();
  }
  
  //---- if number of swaps needed is lesser than the number of threads per block:
  
  if (nswaps>0)
  {
    // swapping nswaps particles (all swaps needed)
    if (threadIdx.x<nswaps)
    {
      p_dummy = p[sh_new_bm[0]+stride];
      p[sh_new_bm[0]+stride] = p[sh_new_bm[1]-stride];
      p[sh_new_bm[1]-stride] = p_dummy;
    }
    __syncthreads();
  }
  
  //---- evaluate new bookmarks and store in global memory
  
  //actualize shared new bookmarks
  if (threadIdx.x == 0)
  {
    if ( (sh_old_bm[0]-sh_new_bm[0]) < (sh_new_bm[1]-sh_old_bm[1]))
    {
      sh_new_bm[1] -= nswaps;
      sh_new_bm[0] = sh_new_bm[1]-1;
    } else
    {
      sh_new_bm[0] += nswaps;
      sh_new_bm[1] = sh_new_bm[0]+1;
    }
  }
  __syncthreads();
  
  // store new bookmarks in global memory
  if (threadIdx.x < 2)
  {
    new_bm[1+blockIdx.x*2+threadIdx.x] = sh_new_bm[threadIdx.x];
  }
  __syncthreads();
  
  return;
}

/**********************************************************/