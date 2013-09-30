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
  
  static const double dtin_e = init_dtin_e();             // time between electron insertions sqrt(2.0*PI*m_e/kT_e)/(n*Lx*dz)
  static const double dtin_i = init_dtin_i();             // time between ion insertions sqrt(2.0*PI*m_i/kT_i)/(n*Lx*dz)
  
  static double tin_e = dtin_e;                           // time for next electron insertion
  static double tin_i = dtin_i;                           // time for next ion insertion

  /*----------------------------- function body -------------------------*/

  particle_cc(t, &tin_e, dtin_e, kte, me, d_e_bm, d_e, d_Ex, d_Ey);
  particle_cc(t, &tin_i, dtin_i, kti, mi, d_i_bm, d_i, d_Ex, d_Ey);
  
  return;
}

/**********************************************************/

void particle_cc(double t, double *tin, double dtin, double kt, double m, int *d_bm, particle **d_p, double *d_Ex, double *d_Ey)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static const double Lx = init_Lx();     //
  static const double Ly = init_Ly();     //
  static const double ds = init_ds();     // geometric properties
  static const int nnx = init_nnx();      // of simulation
  static const int nny = init_nny();      //
  static const int ncy = init_ncy();      //

  cudaError cuError;
  
  // device memory
  int *d_new_bm;      // new particle bookmarks (have to be allocated in device memory)
  
  /*----------------------------- function body -------------------------*/
  
  // allocate device memory for new particle bookmarks
  cuError = cudaMalloc (&d_new_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (d_new_bm, d_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToDevice);
  cu_check(cuError, __FILE__, __LINE__);

  //---- sorting of particles with bining algorithm
  
  particle_bining(Lx, ds, ncy, d_bm, d_new_bm, *d_p);

  //---- apply absorbent/emitter contour conditions
  
  abs_emi_cc(t, tin, dtin, kt, m, d_bm, d_new_bm, d_p, d_Ex, d_Ey);
  
  //---- apply cyclic contour conditions
  
  cyclic_cc(ncy, Lx, d_bm, *d_p);
  
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
  pDefragDown<<<griddim, blockdim>>>(ds, new_bm, p);
  cu_sync_check(__FILE__, __LINE__);
  cudaGetLastError();
  pDefragUp<<<griddim, blockdim>>>(ds, new_bm, p);
  cu_sync_check(__FILE__, __LINE__);
  
  // set dimension of grid of blocks for particle rebracketing kernel
  griddim = ncy-1;
  
  // execute kernel for rebracketing of particles
  cudaGetLastError();
  pRebracketing<<<griddim, blockdim>>>(bm, new_bm, p);
  cu_sync_check(__FILE__, __LINE__);
  //-------------------------------------------------> VOY POR AQUÍ HAY QUE AÑADIR UNA FUNCIÓN QUE MANEJE LOS BOOKMARKS NEGATIVOS
  return;
}

/**********************************************************/

void abs_emi_cc(double t, double *tin, double dtin, double kt, double m, int *d_bm, int *d_new_bm, particle **d_p, double *d_Ex, double *d_Ey)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double Lx = init_Lx();     //
  static const double Ly = init_Ly();     //
  static const double ds = init_ds();     // geometric properties
  static const int nnx = init_nnx();      // of simulation
  static const int nny = init_nny();      //
  static const int ncy = init_ncy();      //
  
  static const double dt = init_dt();     //
  double fpt = t+dt;                      // timing variables
  double fvt = t+0.5*dt;                  //
  
  int in = 0;                             // number of particles added at plasma frontier
  int out_l, out_r;                       // number of particles withdrawn at the probe (l) and the plasma (r)
  
  int h_bm[2*ncy], h_new_bm[2*ncy];       // host particle bookmarks
  int length;                             // length of new particle vectors
  particle *dummy_p;                      // host dummy vector for particle storage
  
  double h_Ex[nnx*nny], h_Ey[nnx*nny];    // host memory for electric fields
  double Epx, Epy;                        // fields at particle position
  int ic, jc;                             // indices of particle cell
  double distx, disty;                    // distance from particle to nodes
  
  static gsl_rng * rng = gsl_rng_alloc(gsl_rng_default);  //default random number generator (gsl)

  cudaError cuError;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // copy new and old bookmark to host memory
  cuError = cudaMemcpy (h_bm, d_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (h_new_bm, d_new_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  cu_check(cuError, __FILE__, __LINE__);
  
  // calculate number of particles that flow into the simulation
  if((*tin) < fpt) in = 1 + int((fpt-(*tin))/dtin);
  
  // calculate number of particles that flow out of the simulation
  out_l = h_new_bm[0]-h_bm[0];
  out_r = h_bm[2*ncy-1]-h_new_bm[2*ncy-1];
  
  // eliminate/create particles
  
  if (out_l != 0 || out_r != 0 || in != 0) {
    // move particles to host dummy vector
    length = h_new_bm[2*ncy-1]-h_new_bm[0]+1+in;
    dummy_p = (particle*) malloc((length)*sizeof(particle));
    cuError = cudaMemcpy(dummy_p, *d_p+h_new_bm[0], (length-in)*sizeof(particle), cudaMemcpyDeviceToHost);
    cu_check(cuError, __FILE__, __LINE__);
    cuError = cudaFree(*d_p);
    cu_check(cuError, __FILE__, __LINE__);
    
    // actualize bookmarks (left removed particles)
    if (out_l != 0) {
      for (int k = 0; k < 2*ncy; k++) {
        h_new_bm[k] -= out_l;
      }
    }
    
    // add particles
    if (in != 0) {
      // copy fields data from device to host for simple push
      cuError = cudaMemcpy (h_Ex, d_Ex, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cu_check(cuError, __FILE__, __LINE__);
      cuError = cudaMemcpy (h_Ey, d_Ey, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cu_check(cuError, __FILE__, __LINE__);
      
      // create new particles
      for (int k = h_new_bm[2*ncy-1]+1; k < length; k++) {
        //initialize particles
        dummy_p[k].x = gsl_rng_uniform_pos(rng)*Lx;
        dummy_p[k].y = Ly;
        dummy_p[k].vx = gsl_ran_gaussian(rng, sqrt(kt/m));
        dummy_p[k].vy = -gsl_ran_rayleigh(rng, sqrt(kt/m));
        
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
        dummy_p[k].x += (fpt-(*tin))*dummy_p[k].vx;
        dummy_p[k].y += (fpt-(*tin))*dummy_p[k].vy;  // comprobar que no atraviesa una celda
        dummy_p[k].vx -= (fvt-(*tin))*Epx/m;
        dummy_p[k].vy -= (fvt-(*tin))*Epy/m;
        
        // actualize time for next particle insertion
        (*tin) += dtin;
      }
      
      // move end bookmark of last bin_bookmark (added particles)
      h_new_bm[2*ncy-1] += in;
    }
    
    // copy new particles to device memory
    cuError = cudaMalloc(d_p, length*sizeof(particle));
    cu_check(cuError, __FILE__, __LINE__);
    cuError = cudaMemcpy(*d_p, dummy_p, length*sizeof(particle), cudaMemcpyHostToDevice);
    cu_check(cuError, __FILE__, __LINE__);
    free(dummy_p);
  }
  
  // copy new bookmarks to device memory
  cuError = cudaMemcpy (d_bm, h_new_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void cyclic_cc(int ncy, double Lx, int *d_bm, particle *d_p)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  
  dim3 griddim, blockdim;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  griddim = ncy;
  blockdim = BINING_BLOCK_DIM;
  
  cudaGetLastError();
  pCyclicCC<<<griddim, blockdim>>>(Lx, d_bm, d_p);
  cu_sync_check(__FILE__, __LINE__);
  
  return;
}

/**********************************************************/


/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void pDefragDown(double ds, int *g_new_bm, particle *g_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ particle sh_p[BINING_BLOCK_DIM];
  __shared__ int sh_bm[2];
  __shared__ int N, tail, i, i_shifted, tpb;
  
  // kernel registers
  int swap_index;
  particle reg_p, tmp_p;
  int tid = (int) threadIdx.x;
  int bid = (int) blockIdx.x;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  
  // load bin bookmarks
  if (tid < 2) sh_bm[tid] = g_new_bm[bid*2+tid];
  __syncthreads();

  // evaluate number of particles in the bin
  if (tid == 0) {
    if (sh_bm[0] >= 0 && sh_bm[1] >= 0) N = sh_bm[1]-sh_bm[0]+1;
    else N = 0;
    tpb = (int) blockDim.x;
  }
  __syncthreads();

  //---- case selection

  if (N > tpb) {
    // initialize batch parameters
    if (0 == tid) {
      while (N < 2*tpb) {
        tpb /= 2;
      }
      tail = 0;
      i = sh_bm[0];
      i_shifted = i + tpb;
      N = 0;
    }
    __syncthreads();
    
    //---- cleanup first batch of "-" particles
    
    // load register batch
    if (tid < tpb) reg_p = g_p[i+tid];
    
    for (int count = 0; N < tpb; count++, __syncthreads()) {
      // load shared memory batch
      if (tid < tpb) sh_p[tid] = g_p[i_shifted+tid+count*tpb];
      __syncthreads();
      // analize register batch
      if (tid < tpb) {
        if (__double2int_rd(reg_p.y/ds) < bid) {
          do {
            swap_index = atomicAdd(&tail, 1);
            if (swap_index >= tpb) break;
          } while (__double2int_rd(sh_p[swap_index].y/ds) < bid);
          if (swap_index >= tpb) continue;
          tmp_p = reg_p;
          reg_p = sh_p[swap_index];
          sh_p[swap_index] = tmp_p;
        }
        atomicAdd(&N, 1);
      }
    }
    
    // store results in global memory
    if (tid < tpb) {
      g_p[i_shifted+tid] = sh_p[tid];
      g_p[i+tid] = reg_p;
    }
    __syncthreads();
    
    // reset tail parameter
    if (0 == tid) tail = 0;
    
    //---- start "-" defrag
    
    while (i_shifted <= sh_bm[1]) {
      // load shared batch
      if (tid < tpb) sh_p[tid] = g_p[i+tid];
      __syncthreads();
      
      // load, analize and swap register batch
      if (tid < tpb) {
        if (i_shifted+tid <= sh_bm[1]) {
          reg_p = g_p[i_shifted+tid];
          if (__double2int_rd(reg_p.y/ds) < bid) {
            swap_index = atomicAdd(&tail, 1);
            tmp_p = reg_p;
            reg_p = sh_p[swap_index];
            sh_p[swap_index] = tmp_p;
          }
        }
      }
      __syncthreads();
      
      // store results in global memory
      if (tid < tpb) {
        g_p[i+tid] = sh_p[tid];
        if (i_shifted+tid <= sh_bm[1]) g_p[i_shifted+tid] = reg_p; 
      }
      
      // actualize batch parameters
      if (0 == tid) {
        i += tail;
        i_shifted += tpb;
        tail = 0;
      }
      __syncthreads();
    }
    
    //---- store new left bookmark in global memory
    
    if (0 == tid) g_new_bm[bid*2] = i;
    
    return;
  }
  else if (N > 0) {
    // load batch parameters
    if (tid == 0) tail = 0;
    __syncthreads();
    
    // load whole bin in registers
    if (tid < N) reg_p = g_p[sh_bm[0]+tid];
    
    // analize and swap whole bin
    if (__double2int_rd(reg_p.y/ds) < bid) {
      swap_index = atomicAdd(&tail, 1);
      sh_p[swap_index] = reg_p;
    }
    __syncthreads();
    
    // store new left bookmark global memory
    if (0 == tid) g_new_bm[bid*2] = sh_bm[0]+tail;
    __syncthreads();
    
    // swap "=" and "+" particles
    if (__double2int_rd(reg_p.y/ds) >= bid) {
      swap_index = atomicAdd(&tail, 1);
      sh_p[swap_index] = reg_p;
    }
    __syncthreads();
    
    // store particle batch in global memory
    if (tid < N) g_p[sh_bm[0]+tid] = sh_p[tid];
    
    return;
  }
  else return;
}

/**********************************************************/

__global__ void pDefragUp(double ds, int *g_new_bm, particle *g_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ particle sh_p[BINING_BLOCK_DIM];
  __shared__ int sh_bm[2];
  __shared__ int N, tail, i, i_shifted, tpb;
  
  // kernel registers
  int swap_index;
  particle reg_p, tmp_p;
  int tid = (int) threadIdx.x;
  int bid = (int) blockIdx.x;
  
  /*--------------------------- kernel body ----------------------------*/

  //---- initialize shared memory
  
  // load bin bookmarks
  if (tid < 2) sh_bm[tid] = g_new_bm[bid*2+tid];
  __syncthreads();
  
  // evaluate number of particles in the bin
  if (tid == 0) {
    if (sh_bm[0] >= 0 && sh_bm[1] >= 0) N = sh_bm[1]-sh_bm[0]+1;
    else N = 0;
    tpb = (int) blockDim.x;
  }
  __syncthreads();
  
  //---- case selection
  
  if (N > tpb) {
    // initialize batch parameters
    if (0 == tid) {
      while (N < 2*tpb) {
        tpb /= 2;
      }
      tail = 0;
      i = sh_bm[1];
      i_shifted = i - tpb;
      N = 0;
    }
    __syncthreads();
    
    //---- cleanup last batch of "+" particles
    
    // load register batch
    if (tid < tpb) reg_p = g_p[i-tid];
    
    for (int count = 0; N < tpb; count++, __syncthreads()) {
      // load shared memory batch
      if (tid < tpb) sh_p[tid] = g_p[i_shifted-tid-count*tpb];
      __syncthreads();
      // analize register batch
      if (tid < tpb) {
        if (__double2int_rd(reg_p.y/ds) > bid) {
          do {
            swap_index = atomicAdd(&tail, 1);
            if (swap_index >= tpb) break;
          } while (__double2int_rd(sh_p[swap_index].y/ds) > bid);
          if (swap_index >= tpb) continue;
          tmp_p = reg_p;
          reg_p = sh_p[swap_index];
          sh_p[swap_index] = tmp_p;
        }
        atomicAdd(&N, 1);
      }
    }
    
    // store results in global memory
    if (tid < tpb) {
      g_p[i_shifted-tid] = sh_p[tid];
      g_p[i-tid] = reg_p;
    }
    __syncthreads();
    
    // reset tail parameter
    if (0 == tid) tail = 0;
    
    //---- start "+" defrag
    
    while (i_shifted >= sh_bm[0]) {
      // load shared batch
      if (tid < tpb) sh_p[tid] = g_p[i-tid];
      __syncthreads();
      
      // load, analize and swap register batch
      if (tid < tpb) {
        if (i_shifted-tid >= sh_bm[0]) {
          reg_p = g_p[i_shifted-tid];
          if (__double2int_rd(reg_p.y/ds) > bid) {
            swap_index = atomicAdd(&tail, 1);
            tmp_p = reg_p;
            reg_p = sh_p[swap_index];
            sh_p[swap_index] = tmp_p;
          }
        }
      }
      __syncthreads();
      
      // store results in global memory
      if (tid < tpb) {
        g_p[i-tid] = sh_p[tid];
        if (i_shifted-tid >= sh_bm[0]) {
          g_p[i_shifted-tid] = reg_p;
        }
      }
      
      // actualize batch parameters
      if (0 == tid) {
        i -= tail;
        i_shifted -= tpb;
        tail = 0;
      }
      __syncthreads();
    }
    
    //---- store new right bookmarks in global memory
    
    if (0 == tid) g_new_bm[bid*2+1] = i;
    
    return;
  }
  else if (N > 0) {
    // load batch parameters
    if (tid == 0) tail = 0;
    __syncthreads();
    
    // load whole bin in registers
    if (tid < N) reg_p = g_p[sh_bm[0]+tid];
    
    // analize and swap whole bin
    if (__double2int_rd(reg_p.y/ds) > bid) {
      swap_index = atomicAdd(&tail, 1);
      sh_p[swap_index] = reg_p;
    }
    __syncthreads();
    
    // store new left bookmark global memory
    if (0 == tid) g_new_bm[bid*2+1] = sh_bm[1]-tail;
    __syncthreads();
    
    // swap "=" particles
    if (__double2int_rd(reg_p.y/ds) = bid) {
      swap_index = atomicAdd(&tail, 1);
      sh_p[swap_index] = reg_p;
    }
    __syncthreads();
    
    // store particle batch in global memory
    if (tid < N) g_p[sh_bm[1]-tid] = sh_p[tid];
    
    return;
  }
  else return;
}

/**********************************************************/

__global__ void pRebracketing(int *bm, int *new_bm, particle *p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_old_bm[2];        // bookmarks before defragmentation (also used to store bookmarks after rebracketing) (bin_end, bin_start)
  __shared__ int sh_new_bm[2];        // bookmarks after particle defragmentation (bin_end, bin_start)
  __shared__ int nswaps;              // number of swaps each bin frontier needs
  // kernel registers
  particle p_dummy;                   // dummy particle for swapping
  int stride = 1 + (int) threadIdx.x; // offset stride for each thread to swap the correct particle
  int tid = (int) threadIdx.x;
  int tpb = (int) blockDim.x;
  int bid = (int) blockIdx.x;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  
  // load old and new bookmarks from global memory
  if (tid < 2) {
    sh_old_bm[tid] = bm[1+bid*2+tid];
    sh_new_bm[tid] = new_bm[1+bid*2+tid];
  }
  __syncthreads();
  
  // evaluate number of swaps needed for each bin frontier
  if (tid == 0) {
    if (sh_old_bm[0] < 0 || sh_old_bm[1] < 0) nswaps = 0;
    else nswaps = (sh_old_bm[0]-sh_new_bm[0])<(sh_new_bm[1]-sh_old_bm[1]) ? (sh_old_bm[0]-sh_new_bm[0]) : (sh_new_bm[1]-sh_old_bm[1]);
  }
  __syncthreads();
  
  //---- swap particles
  
  while (nswaps >= tpb) {
    // swapping of tpb particles
    p_dummy = p[sh_new_bm[0]+stride];
    p[sh_new_bm[0]+stride] = p[sh_new_bm[1]-stride];
    p[sh_new_bm[1]-stride] = p_dummy;
    __syncthreads();
    
    // actualize shared new bookmarks
    if (tid == 0) {
      sh_new_bm[0] += tpb;
      sh_new_bm[1] -= tpb;
      nswaps -= tpb;
    }
    __syncthreads();
  }
  
  if (nswaps>0) {
    if (tid<nswaps) {
      p_dummy = p[sh_new_bm[0]+stride];
      p[sh_new_bm[0]+stride] = p[sh_new_bm[1]-stride];
      p[sh_new_bm[1]-stride] = p_dummy;
    }
    __syncthreads();
  }
  
  //---- actualize new bookmarks and store in global memory
  
  //actualize shared new bookmarks
  if (tid == 0) {
    if (sh_old_bm[0] < 0) {
      if ((sh_new_bm[1]-sh_old_bm[1]) > 0) sh_new_bm[0] = sh_new_bm[1] - 1;
    } else if (sh_old_bm[1] < 0) {
      if ((sh_old_bm[0]-sh_new_bm[0]) > 0) sh_new_bm[1] = sh_new_bm[0] + 1;
    } else {
      if ( (sh_old_bm[0]-sh_new_bm[0]) < (sh_new_bm[1]-sh_old_bm[1])) {
        sh_new_bm[1] -= nswaps;
        sh_new_bm[0] = sh_new_bm[1]-1;
      } else {
        sh_new_bm[0] += nswaps;
        sh_new_bm[1] = sh_new_bm[0]+1;
      }
    }
  }
  __syncthreads();
  
  // store shared new bookmarks in global memory
  if (tid < 2) new_bm[1+bid*2+tid] = sh_new_bm[tid];
  __syncthreads();
  
  return;
}

/**********************************************************/

__global__ void pCyclicCC(double Lx, int *g_bm, particle *g_p)
{
  /*--------------------------- kernel variables -----------------------*/
  
  // kernel shared memory
  __shared__ int sh_bm[2];
  
  // kernel registers
  particle reg_p;
  int tpb = (int) blockDim.x;
  int tid = (int) threadIdx.x;
  int bid = (int) blockIdx.x;
  
  /*--------------------------- kernel body ----------------------------*/
  
  //---- initialize shared memory
  if (tid < 2) {
    sh_bm[tid] = g_bm[tid + bid*2];
  }
  __syncthreads();

  //---- analize cyclic contour condition for every particle in the batch
  for (int i = sh_bm[0]+tid; i <= sh_bm[1]; i += tpb) {
    reg_p = g_p[i];
    if (reg_p.x < 0.0) {
      reg_p.x += Lx;
    } else if (reg_p.x > Lx) {
      reg_p.x -= Lx;
    }
    __syncthreads();
    g_p[i] = reg_p;
  }

  return;
}

/**********************************************************/