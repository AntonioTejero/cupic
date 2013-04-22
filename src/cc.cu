/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "stdh.h"
#include "cc.h"


/************************ FUNCTION DEFINITIONS ***********************/

void particle_bining(double dy, int ncy, int *bookmark, int *new_bookmark, particle **p)
{
  /*--------------------------- function variables -----------------------*/

  dim3 griddim, blockdim;

  /*----------------------------- function body -------------------------*/

  // set dimensions of grid of blocks and blocks of threads for particle defragmentation kernel
  griddim = ncy;
  blockdim = BINING_BLOCK_DIM;

  // execute kernel for defragmentation of particles
  particle_defragmentation<<<griddim, blockdim>>>(bookmark, new_bookmark, dy, *p);

  // set dimension of grid of blocks for particle rebracketing kernel
  griddim = ncy-1;

  // execute kernel for rebracketing of particles
  particle_rebracketing<<<griddim, blockdim>>>(bookmark, new_bookmark, *p);

  return;
}

/**********************************************************/

__global__ void particle_defragmentation(int *bookmark, int *new_bookmark, double dy, particle *p)
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

  // obtaining valid swap_index for each "-" particle in first batch
  new_bin = p_reg.y/dy;
  if (new_bin<bin)
  {
    do
    {
      swap_index = atomicAdd(&tail, 1);
    } while (int(p_sha[swap_index].y/dy)<bin);
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

      // analyze batch of particle in registers
      new_bin = p_reg.y/dy;
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
  new_bin = p_reg.y/dy;
  if (new_bin>bin)
  {
    do
    {
      swap_index = atomicAdd(&tail, 1);
    } while (int(p_sha[swap_index].y/dy)>bin);
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
      new_bin = p_reg.y/dy;
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

__global__ void particle_rebracketing(int *bookmark, int *new_bookmark, particle *p)
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

// void cc_abs_inj (double t, double dt int *e_bookmark, int *e_new_bookmark, particle **e, int *i_bookmark, int * i_new_bookmark, particle **i)
// {
//   /*--------------------------- function variables -----------------------*/
// 
//   double delta_e = 1.0e-1;    //time between electron insertions
//   double delta_i = 1.0e0;     //time between ion insertions
//   int n_e;                    //number of electron added at plasma frontier
//   int n_i;                    //number of ions added at plasma frontier
// 
//   /*----------------------------- function body -------------------------*/
// 
//   // calculate number of electrons and ions that flow into the simulation
//   n_e = t
// 
// 
// 
//   return;
// }

/**********************************************************/