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

void cc (double t, unsigned int *d_e_bm, particle **d_e, unsigned int *d_i_bm, particle **d_i, double *d_Ex, double *d_Ey)
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
  static const int ncy = init_ncx();                      // number of cells in y dimension
  
  static double tin_e = dtin_e;                           // time for next electron insertion
  static double tin_i = dtin_i;                           // time for next ion insertion
  double fpt = t+dt;                                      // future position time
  double fvt = t+0.5*dt;                                  // future velocity time
  int in_e, in_i;                                         // number of electron and ions added at plasma frontier
  int out_e_l, out_e_r, out_i_l, out_i_r;                 // number of electrons and ions withdrawn at probe (l) and at plasma frontier (r)
  
  unsigned int h_e_bm[2*ncy], h_i_bm[2*ncy];              // old particle bookmarks
  unsigned int h_e_new_bm[2*ncy], h_i_new_bm[2*ncy];      // new particle bookmarks
  int length;                                             // length of particle vectors
  
  double *h_Ex, *h_Ey;                                    // host memory for electric fields
  double Epx, Epy;                                        // fields at particle position
  int ic, jc;                                             // indices of particle cell
  double distx, disty;                                    // distance from particle to nodes  
  particle *dummy_p;                                      // dummy vector for particle storage
  
  static gsl_rng * rng = gsl_rng_alloc(gsl_rng_default);  //default random number generator (gsl)
  
  // device memory
  unsigned int *d_e_new_bm, *d_i_new_bm;      // new particle bookmarks (have to be allocated in device memory)

  /*----------------------------- function body -------------------------*/

  //---- sorting and cyclic contour conditions
  
  // allocate device memory for new particle bookmarks
  cudaMalloc (&d_e_new_bm, 2*ncy*sizeof(unsigned int));
  cudaMalloc (&d_i_new_bm, 2*ncy*sizeof(unsigned int));  
  
  // sort particles with bining algorithm, also apply cyclic contour conditions during particle defragmentation
  particle_bining(Lx, ds, ncy, d_e_bm, d_e_new_bm, *d_e);
  particle_bining(Lx, ds, ncy, d_i_bm, d_i_new_bm, *d_i);
  
  // copy new and old bookmark to host memory
  cudaMemcpy (h_e_bm, d_e_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_i_bm, d_i_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_e_new_bm, d_e_new_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_i_new_bm, d_i_new_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
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
    length = h_e_new_bm[2*ncy-1]-h_e_new_bm[0]+1;                                             // calculate number of particles that remains
    dummy_p = (particle*) malloc((length+in_e)*sizeof(particle));                             // allocate intermediate particle vector in host memory
    cudaMemcpy(dummy_p, *d_e+h_e_new_bm[0], length*sizeof(particle), cudaMemcpyDeviceToHost); // move remaining particles to dummy vector (host memory)
    cudaFree(*d_e);                                                                           // free old particles device memory
    
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
      cudaMemcpy (h_Ex, d_Ex, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy (h_Ey, d_Ey, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      
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
        dummy_p[k].y += (fpt-tin_e)*dummy_p[k].vy;
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
    cudaMalloc(d_e, length*sizeof(particle));                                   // allocate new device memory for particles
    cudaMemcpy(*d_e, dummy_p, length*sizeof(particle), cudaMemcpyHostToDevice); // copy new particles to device memory
    free(dummy_p);                                                              // free intermediate particle vector (host memory)
  }
  
  //-- ions
  if (out_i_l != 0 || out_i_r != 0 || in_i != 0)
  {
    // move particles to host dummy vector
    length = h_i_new_bm[2*ncy-1]-h_i_new_bm[0]+1;                                             // calculate number of particles that remains
    dummy_p = (particle*) malloc((length+in_i)*sizeof(particle));                             // allocate intermediate particle vector in host memory
    cudaMemcpy(dummy_p, *d_i+h_i_new_bm[0], length*sizeof(particle), cudaMemcpyDeviceToHost); // move remaining particles to dummy vector (host memory)
    cudaFree(*d_i);                                                                           // free old particles device memory
    
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
      cudaMemcpy (h_Ex, d_Ex, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      cudaMemcpy (h_Ey, d_Ey, nnx*nny*sizeof(double), cudaMemcpyDeviceToHost);
      
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
        dummy_p[k].y += (fpt-tin_i)*dummy_p[k].vy;
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
    cudaMalloc(d_i, length*sizeof(particle));                                   // allocate new device memory for particles
    cudaMemcpy(*d_i, dummy_p, length*sizeof(particle), cudaMemcpyHostToDevice); // copy new particles to device memory
    free(dummy_p);                                                              // free intermediate particle vector (host memory)
  }
  
  // copy new bookmarks to device memory
  cudaMemcpy (d_e_bm, h_e_new_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (d_i_bm, h_i_new_bm, 2*ncy*sizeof(unsigned int), cudaMemcpyHostToDevice);
  
  return;
}

/**********************************************************/

inline void particle_bining(double Lx, double ds, int ncy, unsigned int *bookmark, unsigned int *new_bookmark, particle *p)
{
  /*--------------------------- function variables -----------------------*/

  dim3 griddim, blockdim;

  /*----------------------------- function body --------------------------*/

  // set dimensions of grid of blocks and blocks of threads for particle defragmentation kernel
  griddim = ncy;
  blockdim = BINING_BLOCK_DIM;

  // execute kernel for defragmentation of particles
  particle_defragmentation<<<griddim, blockdim>>>(Lx, ds, bookmark, new_bookmark, p);

  // set dimension of grid of blocks for particle rebracketing kernel
  griddim = ncy-1;

  // execute kernel for rebracketing of particles
  particle_rebracketing<<<griddim, blockdim>>>(bookmark, new_bookmark, p);

  return;
}

/**********************************************************/


/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void particle_defragmentation(double Lx, double ds, unsigned int *bookmark, unsigned int *new_bookmark, particle *p)
{
  /*--------------------------- kernel variables -----------------------*/

  // kernel shared memory
  __shared__ particle p_sha[BINING_BLOCK_DIM];
  __shared__ int bin;
  __shared__ int bin_bookmark[2];
  __shared__ int tail, i, i_shifted;
  // kernel registers
  int new_bin, swap_index;
  particle p_reg, p_dummy;

  /*--------------------------- kernel body ----------------------------*/

  //---- initialize shared memory

  // initialize bin variable (the same as blockIdx.x)
  if (threadIdx.x == 0) bin = blockIdx.x;

  // load bin bookmarks
  if (threadIdx.x < 2)
  {
    bin_bookmark[threadIdx.x] = bookmark[bin*2+threadIdx.x];
  }

  // initialize batches and tail parameters for "-" defrag algorithm
  if (threadIdx.x == 0)
  {
    i = bin_bookmark[0];
    i_shifted = i + blockDim.x;
    tail = 0;
  }
  __syncthreads();

  //---- cleaning first batch of particles

  // reading from global memory
  p_sha[threadIdx.x] = p[i_shifted+threadIdx.x];
  __syncthreads();
  p_reg = p[i+threadIdx.x];
  
  // apply cyclic contour condition
  if (p_reg.x<0)
  {
    p_reg.x += Lx;  
  } else if (p_reg.x>Lx)
  {
    p_reg.x -= Lx;
  }

  // obtaining valid swap_index for each "-" particle in first batch
  new_bin = p_reg.y/ds;
  if (new_bin<bin)
  {
    do
    {
      swap_index = atomicAdd(&tail, 1);
    } while (int(p_sha[swap_index].y/ds)<bin);
    // swapping "-" particles from first batch with "non -" particles from second batch
    p_dummy = p_reg;
    p_reg = p_sha[swap_index];
    p_sha[swap_index] = p_dummy;
  }
  __syncthreads();

  // write back particle batches to global memory
  p[i+threadIdx.x] = p_reg;
  __syncthreads();
  p[i_shifted+threadIdx.x] = p_sha[threadIdx.x];

  // reset tail parameter (shared memory)
  if (threadIdx.x == 0)
  {
    tail = 0;
  }
  __syncthreads();

  //---- start of "-" defrag algorithm

  while (i_shifted<=bin_bookmark[1])
  {
    // copy exchange queue from global memory to shared memory
    p_sha[threadIdx.x] = p[i+threadIdx.x];
    __syncthreads();

    if (i_shifted+threadIdx.x<=bin_bookmark[1])
    {
      // copy batch of particles to be analyzed from global memory to registers
      p_reg = p[i_shifted+threadIdx.x];

      // apply cyclic contour condition
      if (p_reg.x<0)
      {
        p_reg.x += Lx;  
      } else if (p_reg.x>Lx)
      {
        p_reg.x -= Lx;
      } 

      // analyze batch of particle in registers
      new_bin = p_reg.y/ds;
      if (new_bin<bin)
      {
        // swapping "-" particles from registers with particles in exchange queue (shared memory)
        swap_index = atomicAdd(&tail, 1);
        p_dummy = p_reg;
        p_reg = p_sha[swap_index];
        p_sha[swap_index] = p_dummy;
      }
    }
    __syncthreads();

    // write back particle batches to global memory
    if (i_shifted+threadIdx.x<=bin_bookmark[1])
    {
      p[i_shifted+threadIdx.x] = p_reg;
    }
    __syncthreads();
    p[i+threadIdx.x] = p_sha[threadIdx.x];
    __syncthreads();

    // actualize parameters in shared memory
    if (threadIdx.x == 0)
    {
      // update batches parameters for next iteration (shared memory)
      i += tail;
      i_shifted += blockDim.x;
      // reset tail parameter (shared memory)
      tail = 0;
    }
  }

  // actualize bin_bookmark to the new "bin_start" value
  if (threadIdx.x == 0)
  {
    bin_bookmark[0] = i;
  }
  __syncthreads();

  //---- reset shared memory variables for "+" defrag algorithm

  if (threadIdx.x == 0)
  {
    i = bin_bookmark[1];
    i_shifted = i - blockDim.x;
    tail = 0;
  }
  __syncthreads();

  //---- cleaning last batch of particles

  // reading from global memory
  p_sha[threadIdx.x] = p[i_shifted-threadIdx.x];
  __syncthreads();
  p_reg = p[i-threadIdx.x];

  // obtaining valid swap_index for each "+" particle in last batch
  new_bin = p_reg.y/ds;
  if (new_bin>bin)
  {
    do
    {
      swap_index = atomicAdd(&tail, 1);
    } while (int(p_sha[swap_index].y/ds)>bin);
    // swapping "+" particles from first batch with "non +" particles from second batch
    p_dummy = p_reg;
    p_reg = p_sha[swap_index];
    p_sha[swap_index] = p_dummy;
  }
  __syncthreads();

  // write back particle batches to global memory
  p[i-threadIdx.x] = p_reg;
  __syncthreads();
  p[i_shifted-threadIdx.x] = p_sha[threadIdx.x];
  __syncthreads();

  // reset tail parameter (shared memory)
  if (threadIdx.x ==0)
  {
    tail = 0;
  }

  //---- start of "+" defrag algorithm

  while (i_shifted>=bin_bookmark[0])
  {
    // copy exchange queue from global memory to shared memory
    p_sha[threadIdx.x] = p[i-threadIdx.x];
    __syncthreads();

    if (i_shifted-threadIdx.x>=bin_bookmark[0])
    {
      // copy batch of particles to be analyzed from global memory to registers
      p_reg = p[i_shifted-threadIdx.x];

      // analyze batch of particle in registers
      new_bin = p_reg.y/ds;
      if (new_bin>bin)
      {
        // swapping "+" particles from registers with particles in exchange queue (shared memory)
        swap_index = atomicAdd(&tail, 1);
        p_dummy = p_reg;
        p_reg = p_sha[swap_index];
        p_sha[swap_index] = p_dummy;
      }
    }
    __syncthreads();

    // write back particle batches to global memory
    if (i_shifted-threadIdx.x>=bin_bookmark[0])
    {
      p[i_shifted-threadIdx.x] = p_reg;
    }
    __syncthreads();
    p[i-threadIdx.x] = p_sha[threadIdx.x];
    __syncthreads();

    // actualize parameters in shared memory
    if (threadIdx.x == 0)
    {
      // update batches parameters for next iteration (shared memory)
      i -= tail;
      i_shifted -= blockDim.x;
      // reset tail parameter (shared memory)
      tail = 0;
    }
  }

  // actualize bin_bookmark to the new "bin_end" value
  if (threadIdx.x == 0)
  {
    bin_bookmark[1] = i;
  }
  __syncthreads();

  //---- store actualized bookmarks in new bin bookmarks variable in global memory

  if (threadIdx.x < 2)
  {
    new_bookmark[bin*2+threadIdx.x] = bin_bookmark[threadIdx.x];
  }

  return;
}

/**********************************************************/

__global__ void particle_rebracketing(unsigned int *bookmark, unsigned int *new_bookmark, particle *p)
{
  /*--------------------------- kernel variables -----------------------*/

  // kernel shared memory
  __shared__ int sh_old_bookmark[2];  // bookmarks before defragmentation (also used to store bookmarks after rebracketing) (bin_end, bin_start)
  __shared__ int sh_new_bookmark[2];  // bookmarks after particle defragmentation (bin_end, bin_start)
  __shared__ int nswaps;              // number of swaps each bin frontier needs
  __shared__ int tpb;                 // threads per block
  // kernel registers
  particle p_dummy;                   // dummy particle for swapping
  int stride = 1+threadIdx.x;         // offset stride thath of each thread to swap the correct particle


  /*--------------------------- kernel body ----------------------------*/

  //---- initialize shared memory

  // load old and new bookmarks from global memory
  if (threadIdx.x < 2)
  {
    sh_old_bookmark[threadIdx.x] = bookmark[1+blockIdx.x*2+threadIdx.x];
    sh_new_bookmark[threadIdx.x] = new_bookmark[1+blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();

  // set tpb variable and evaluate number of swaps needed for each bin frontier
  if (threadIdx.x == 0)
  {
    tpb = blockDim.x;
    nswaps = ( (sh_old_bookmark[0]-sh_new_bookmark[0])<(sh_new_bookmark[1]-sh_old_bookmark[1]) ) ? (sh_old_bookmark[0]-sh_new_bookmark[0]) : (sh_new_bookmark[1]-sh_old_bookmark[1]);
  }
  __syncthreads();

  //---- if number of swaps needed is greater than the number of threads per block:

  while (nswaps >= tpb)
  {
    // swapping of tpb particles
    p_dummy = p[sh_new_bookmark[0]+stride];
    p[sh_new_bookmark[0]+stride] = p[sh_new_bookmark[1]-stride];
    p[sh_new_bookmark[1]-stride] = p_dummy;
    __syncthreads();

    // actualize shared new bookmarks
    if (threadIdx.x == 0)
    {
      sh_new_bookmark[0] += tpb;
      sh_new_bookmark[1] -= tpb;
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
      p_dummy = p[sh_new_bookmark[0]+stride];
      p[sh_new_bookmark[0]+stride] = p[sh_new_bookmark[1]-stride];
      p[sh_new_bookmark[1]-stride] = p_dummy;
    }
    __syncthreads();
  }

  //---- evaluate new bookmarks and store in global memory

  //actualize shared new bookmarks
  if (threadIdx.x == 0)
  {
    if ( (sh_old_bookmark[0]-sh_new_bookmark[0]) < (sh_new_bookmark[1]-sh_old_bookmark[1]))
    {
      sh_new_bookmark[1] -= nswaps;
      sh_new_bookmark[0] = sh_new_bookmark[1]-1;
    } else
    {
      sh_new_bookmark[0] += nswaps;
      sh_new_bookmark[1] = sh_new_bookmark[0]+1;
    }
  }
  __syncthreads();

  // store new bookmarks in global memory
  if (threadIdx.x < 2)
  {
    new_bookmark[1+blockIdx.x*2+threadIdx.x] = sh_new_bookmark[threadIdx.x];
  }
  __syncthreads();

  return;
}

/**********************************************************/