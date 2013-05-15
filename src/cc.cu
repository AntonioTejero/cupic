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

void cc (double t, double dt, double Lx, double ds, int ncy, unsigned int *d_e_bookmark, particle **e, unsigned int *d_i_bookmark, particle **i)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  static double delta_e = 1.0e-1;     //time between electron insertions sqrt(2.0*PI*m_e/kT_e)/(n*Lx*dz)
  static double delta_i = 1.0e0;      //time between ion insertions
  static double lt_e = 0.0;           //last time an electron where introduced
  static double lt_i = 0.0;           //last time an ion where introduced
  int in_e, in_i;                     //number of electron and ions added at plasma frontier
  int out_e, out_i;                   //number of electrons and ions a at plasma frontier
  
  unsigned int h_e_bookmark[2*ncy], h_i_bookmark[2*ncy];           //old particle bookmarks
  unsigned int h_e_new_bookmark[2*ncy], h_i_new_bookmark[2*ncy];   //new particle bookmarks
  
  // device memory
  unsigned int *d_e_new_bookmark, *d_i_new_bookmark;   //new particle bookmarks (have to be allocated in device memory)
  particle *dummy_p;                                   //dummy vector for particle storage

  /*----------------------------- function body -------------------------*/

  //---- sorting and cyclic contour conditions
  
  // calculate number of electrons and ions that flow into/out to the simulation
  in_e = (t+dt-lt_e)/delta_e;
  in_i = (t+dt-lt_i)/delta_i;
  
  // allocate device memory for new particle bookmarks
  cudaMalloc (&d_e_new_bookmark, 2*ncy*sizeof(unsigned int));
  cudaMalloc (&d_i_new_bookmark, 2*ncy*sizeof(unsigned int));  
  
  // sort particles with bining algorithm, also apply cyclic contour conditions during particle defragmentation
  particle_bining(Lx, ds, ncy, d_e_bookmark, d_e_new_bookmark, *e);
  particle_bining(Lx, ds, ncy, d_i_bookmark, d_i_new_bookmark, *i);
  
  // copy new bookmark to host memory
  cudaMemcpy (h_e_bookmark, d_e_bookmark, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_i_bookmark, d_i_bookmark, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_e_new_bookmark, d_e_new_bookmark, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  cudaMemcpy (h_i_new_bookmark, d_i_new_bookmark, 2*ncy*sizeof(unsigned int), cudaMemcpyDeviceToHost);
  
  // calculate outflowing particles of each type
  out_e = h_e_new_bookmark[0]-h_e_bookmark[0]+h_e_bookmark[2*ncy-1]-h_e_new_bookmark[2*ncy-1];
  out_i = h_i_new_bookmark[0]-h_i_bookmark[0]+h_i_bookmark[2*ncy-1]-h_i_new_bookmark[2*ncy-1];
  
  //---- absorbent/emitter contour conditions
  
  // electrons
  if (out_e != in_e) 
  {
    int length = h_e_new_bookmark[2*ncy-1]-h_e_new_bookmark[0]+1;                                   // calculate number of particles that remains
    dummy_p = (particle*) malloc((length+in_e)*sizeof(particle));                                   // allocate intermediate particle vector in host memory
    cudaMemcpy(dummy_p, *e+h_e_new_bookmark[0], length*sizeof(particle), cudaMemcpyDeviceToHost);   // move remaining particles to dummy vector (host memory)
    
    // FIELDS NEEDED FOR SIMPLE PUSH (INSERTION OF PARTICLES NEED TO BE IMPLEMENTED)
    
    cudaFree(*e);                                                                                   // free old particles device memory
    cudaMalloc(e, (length+in_e)*sizeof(particle));                                                  // allocate new device memory for particles
    cudaMemcpy(*e, dummy_p, (length+in_e)*sizeof(particle), cudaMemcpyHostToDevice);                // copy new particles to device memory
    free(dummy_p);                                                                                  // free intermediate particle vector (host memory)
  } else
  {
    // FIELDS NEEDED FOR SIMPLE PUSH (INSERTION OF PARTICLES NEED TO BE IMPLEMENTED)
  }
  
  // ions
  if (out_i != in_i) 
  {
    int length = h_i_new_bookmark[2*ncy-1]-h_i_new_bookmark[0]+1;                                   // calculate number of particles that remains
    dummy_p = (particle*) malloc((length+in_i)*sizeof(particle));                                   // allocate intermediate particle vector in host memory
    cudaMemcpy(dummy_p, i+h_i_new_bookmark[0], length*sizeof(particle), cudaMemcpyDeviceToHost);    // move remaining particles to dummy vector (host memory)
    
    // FIELDS NEEDED FOR SIMPLE PUSH (INSERTION OF PARTICLES NEED TO BE IMPLEMENTED)
    
    cudaFree(i);                                                                                    // free old particles device memory
    cudaMalloc(i, (length+in_i)*sizeof(particle));                                                  // allocate new device memory for particles
    cudaMemcpy(i, dummy_p, (length+in_e)*sizeof(particle), cudaMemcpyHostToDevice);                 // copy new particles to device memory
    free(dummy_p);                                                                                  // free intermediate particle vector (host memory)
  } else
  {
    // FIELDS NEEDED FOR SIMPLE PUSH (INSERTION OF PARTICLES NEED TO BE IMPLEMENTED)
  }
  
  return;
}

/**********************************************************/

void particle_bining(double Lx, double ds, int ncy, unsigned int *bookmark, unsigned int *new_bookmark, particle *p)
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