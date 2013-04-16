/****************************************************************************
 *                                                                          *
 *    CUPIC is a code that simulates the interaction between plasma and     *
 *    a langmuir probe using PIC techniques accelerated with the use of     *
 *    GPU hardware (CUDA extension of C/C++)                                *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <gsl/gsl_rng.h>        //gsl library for random number generation
#include <gsl/gsl_randist.h>    //gsl library for random number generation

using namespace std;

#define PI 3.1415926535897932   //symbolic constant for PI
#define BINING_BLOCK_DIM 1024   //block dimension for defragmentation kernel

struct particle
{
  double x;
  double y;
  double vx;
  double vy;
};

/****************************** FUNCTION PROTOTIPES ******************************/

void initialize (double **h_qi, double **h_qe, double **h_mi, double **h_me, double **h_kti, double **h_kte, double **h_phi_p, double **h_n, double **h_Lx, double **h_Ly, double **h_dx, double **h_dy, double **h_dz, double **h_t, double **h_dt, double **h_epsilon, double **h_rho, double **h_phi, double **h_Ex, double **h_Ey, particle **h_e, particle **h_i, unsigned int **h_bookmarke, unsigned int **h_bookmarki, double **d_qi, double **d_qe, double **d_mi, double **d_me, double **d_kti, double **d_kte, double **d_phi_p, double **d_n, double **d_Lx, double **d_Ly, double **d_dx, double **d_dy, double **d_dz, double **d_t, double **d_dt, double **d_epsilon, double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, unsigned int **d_bookmarke, unsigned int **d_bookmarki);

void read_input_file (double *h_qi, double *h_qe, double *h_mi, double *h_me, double *h_kti, double *h_kte, double *h_phi_p, double *h_n, double *h_Lx, double *h_Ly, double *h_dx, double *h_dy, double *h_dz, double *h_t, double *h_dt, double *h_epsilon);

// kernels
__global__ void particle_defragmentation(int *bookmark, int *new_bookmark, double dy, particle *p);
__global__ void particle_rebracketing(int offset, int *bookmark, int *new_bookmark, particle *p);
/****************************** MAIN FUNCTION ******************************/

int main (int argc, const char* argv[])
{
  // host variables definition
  double *h_qi, *h_qe, *h_mi, *h_me, *h_kti, *h_kte;  //properties of particles (charge, mass and temperature of particle species)
  double *h_n;                                        //plasma properties (plasma density)
  double *h_phi_p;                                    //probe properties (probe potential)
  double *h_Lx, *h_Ly, *h_dx, *h_dy, *h_dz;           //geometrical properties of simulation (simulation dimensions and spacial step)
  double *h_epsilon;                                  //electromagnetic properties
  double *h_rho, *h_phi, *h_Ex, *h_Ey;                //properties of mesh (charge density, potential and fields at point of the mesh)
  double *h_t, *h_dt;                                 //timing variables (simulation time and time step)
  particle *h_e, *h_i;                                //vector of electron and ions
  unsigned int *h_bookmarke;                          //vector that stores the endpoint of each particle bin (electrons)
  unsigned int *h_bookmarki;                          //vector that stores the endpoint of each particle bin (ions)

  // device variables definition
  double *d_qi, *d_qe, *d_mi, *d_me, *d_kti, *d_kte;  //properties of particles (charge, mass and temperature of particle species)
  double *d_n;                                        //plasma properties (plasma density)
  double *d_phi_p;                                    //probe properties (probe potential)
  double *d_Lx, *d_Ly, *d_dx, *d_dy, *d_dz;           //geometrical properties of simulation (simulation dimensions and spacial step)
  double *d_epsilon;                                  //electromagnetic properties
  double *d_rho, *d_phi, *d_Ex, *d_Ey;                //properties of mesh (charge density, potential and fields at point of the mesh)
  double *d_t, *d_dt;                                 //timing variables (simulation time and time step)
  particle *d_e, *d_i;                                //vector of electron and ions
  unsigned int *d_bookmarke;                          //vector that stores the endpoint of each particle bin (electrons)
  unsigned int *d_bookmarki;                          //vector that stores the endpoint of each particle bin (ions)

  initialize (&h_qi, &h_qe, &h_mi, &h_me, &h_kti, &h_kte, &h_phi_p, &h_n, &h_Lx, &h_Ly, &h_dx, &h_dy, &h_dz, &h_t, &h_dt, &h_epsilon, &h_rho, &h_phi, &h_Ex, &h_Ey, &h_e, &h_i, &h_bookmarke, &h_bookmarki, &d_qi, &d_qe, &d_mi, &d_me, &d_kti, &d_kte, &d_phi_p, &d_n, &d_Lx, &d_Ly, &d_dx, &d_dy, &d_dz, &d_t, &d_dt, &d_epsilon, &d_rho, &d_phi, &d_Ex, &d_Ey, &d_e, &d_i, &d_bookmarke, &d_bookmarki);

  return 0;
}

/****************************** FUNCTION DEFINITION ******************************/

void initialize (double **h_qi, double **h_qe, double **h_mi, double **h_me, double **h_kti, double **h_kte, double **h_phi_p, double **h_n, double **h_Lx, double **h_Ly, double **h_dx, double **h_dy, double **h_dz, double **h_t, double **h_dt, double **h_epsilon, double **h_rho, double **h_phi, double **h_Ex, double **h_Ey, particle **h_e, particle **h_i, unsigned int **h_bookmarke, unsigned int **h_bookmarki, double **d_qi, double **d_qe, double **d_mi, double **d_me, double **d_kti, double **d_kte, double **d_phi_p, double **d_n, double **d_Lx, double **d_Ly, double **d_dx, double **d_dy, double **d_dz, double **d_t, double **d_dt, double **d_epsilon, double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, unsigned int **d_bookmarke, unsigned int **d_bookmarki)
{
  // function variables
  int N;                                          //initial number of particle of each species
  int ncx, ncy;                                   //number of grid points in each dimension
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_default); //default random number generator (gsl)

  // initialize enviromental variables for gsl random number generator
  gsl_rng_env_setup();

  // allocate host memory for particle properties
  *h_qi = (double*) malloc(sizeof(double));
  *h_qe = (double*) malloc(sizeof(double));
  *h_mi = (double*) malloc(sizeof(double));
  *h_me = (double*) malloc(sizeof(double));
  *h_kti = (double*) malloc(sizeof(double));
  *h_kte = (double*) malloc(sizeof(double));

  // allocate host memory for plasma properties
  *h_n = (double*) malloc(sizeof(double));

  // allocate host memory for probe properties
  *h_phi_p = (double*) malloc(sizeof(double));

  // allocate host memory for geometrical properties of simulation
  *h_Lx = (double*) malloc(sizeof(double));
  *h_Ly = (double*) malloc(sizeof(double));
  *h_dx = (double*) malloc(sizeof(double));
  *h_dy = (double*) malloc(sizeof(double));
  *h_dz = (double*) malloc(sizeof(double));

  // allocate host memory for electromagnetic properties
  *h_epsilon = (double*) malloc(sizeof(double));

  // allocate host memory for mesh properties
  *h_rho = (double*) malloc(sizeof(double));
  *h_phi = (double*) malloc(sizeof(double));
  *h_Ex = (double*) malloc(sizeof(double));
  *h_Ey = (double*) malloc(sizeof(double));

  // allocate host memory for timing variables
  *h_t = (double*) malloc(sizeof(double));
  *h_dt = (double*) malloc(sizeof(double));

  // allocate device memory for particle properties
  cudaMalloc (d_qi, sizeof(double));
  cudaMalloc (d_qe, sizeof(double));
  cudaMalloc (d_mi, sizeof(double));
  cudaMalloc (d_me, sizeof(double));
  cudaMalloc (d_kti, sizeof(double));
  cudaMalloc (d_kte, sizeof(double));

  // allocate device memory for plasma properties
  cudaMalloc (d_n, sizeof(double));

  // allocate device memory for probe properties
  cudaMalloc (d_phi_p, sizeof(double));

  // allocate device memory for geometrical properties of simulation
  cudaMalloc (d_Lx, sizeof(double));
  cudaMalloc (d_Ly, sizeof(double));
  cudaMalloc (d_dx, sizeof(double));
  cudaMalloc (d_dy, sizeof(double));
  cudaMalloc (d_dz, sizeof(double));

  // allocate device memory for electromagnetic properties
  cudaMalloc (d_epsilon, sizeof(double));

  // allocate device memory for mesh properties
  cudaMalloc (d_rho, sizeof(double));
  cudaMalloc (d_phi, sizeof(double));
  cudaMalloc (d_Ex, sizeof(double));
  cudaMalloc (d_Ey, sizeof(double));

  // allocate device memory for timing variables
  cudaMalloc (d_t, sizeof(double));
  cudaMalloc (d_dt, sizeof(double));

  // read input file
  read_input_file (*h_qi, *h_qe, *h_mi, *h_me, *h_kti, *h_kte, *h_phi_p, *h_n, *h_Lx, *h_Ly, *h_dx, *h_dy, *h_dz, *h_t, *h_dt, *h_epsilon);

  // calculate initial number of particles and number of mesh points
  N = (**h_Lx)*(**h_dy)*(**h_dz)*(**h_n);
  ncx = (**h_Lx)/(**h_dx)+1;
  ncy = (**h_Ly)/(**h_dy)+1;
  N *= ncy;

  // allocate host memory for particle vectors
  *h_i = (particle*) malloc(N*sizeof(particle));
  *h_e = (particle*) malloc(N*sizeof(particle));

  // allocate host memory for bookmark vectors
  *h_bookmarke = (unsigned int*) malloc((ncy-1)*sizeof(unsigned int));
  *h_bookmarki = (unsigned int*) malloc((ncy-1)*sizeof(unsigned int));

  // allocate host memory for mesh variables
  *h_rho = (double*) malloc(ncx*ncy*sizeof(double));
  *h_phi = (double*) malloc(ncx*ncy*sizeof(double));
  *h_Ex = (double*) malloc(ncx*ncy*sizeof(double));
  *h_Ey = (double*) malloc(ncx*ncy*sizeof(double));

  // allocate device memory for particle vectors
  cudaMalloc (d_i, N*sizeof(particle));
  cudaMalloc (d_e, N*sizeof(particle));

  // allocate device memory for bookmark vectors
  cudaMalloc (d_bookmarke, (ncy-1)*sizeof(unsigned int));
  cudaMalloc (d_bookmarki, (ncy-1)*sizeof(unsigned int));

  // allocate device memory for mesh variables
  cudaMalloc (d_rho, ncx*ncy*sizeof(double));
  cudaMalloc (d_phi, ncx*ncy*sizeof(double));
  cudaMalloc (d_Ex, ncx*ncy*sizeof(double));
  cudaMalloc (d_Ey, ncx*ncy*sizeof(double));

  // initialize particle vectors and bookmarks (host memory)
  for (int i = 0; i < ncy-1; i++)
  {
    (*h_bookmarke)[i] = (i+1)*N/(ncy-1);
    (*h_bookmarki)[i] = (i+1)*N/(ncy-1);
    for (int j = 0; j < N/(ncy-1); j++)
    {
      // initialize ions
      (*h_i)[i*N/(ncy-1)+j].x = gsl_rng_uniform_pos(rng)*(**h_Lx);
      (*h_i)[i*N/(ncy-1)+j].y = double(i)*(**h_dy)+gsl_rng_uniform_pos(rng)*(**h_dy);
      (*h_i)[i*N/(ncy-1)+j].vx = gsl_ran_gaussian(rng, sqrt((**h_kti)/(**h_mi)));
      (*h_i)[i*N/(ncy-1)+j].vy = gsl_ran_gaussian(rng, sqrt((**h_kti)/(**h_mi)));

      // initialize electrons
      (*h_e)[i*N/(ncy-1)+j].x = gsl_rng_uniform_pos(rng)*(**h_Lx);
      (*h_e)[i*N/(ncy-1)+j].y = double(i)*(**h_dy)+gsl_rng_uniform_pos(rng)*(**h_dy);
      (*h_e)[i*N/(ncy-1)+j].vx = gsl_ran_gaussian(rng, sqrt((**h_kte)/(**h_me)));
      (*h_e)[i*N/(ncy-1)+j].vy = gsl_ran_gaussian(rng, sqrt((**h_kte)/(**h_me)));
    }
  }

  //initialize mesh variables (host memory)
  for (int im = 0; im < ncx; im++)
  {
    for (int jm = 0; jm < ncy; jm++)
    {
      (*h_Ex)[im+jm*(ncx)] = 0.0;
      (*h_Ey)[im+jm*(ncx)] = 0.0;
      (*h_rho)[im+jm*(ncx)] = 0.0;
      (*h_phi)[im+jm*(ncx)] = (1.0 - double(jm)/double(ncy-1))*(**h_phi_p);
    }
  }

  // copy particle properties from host to device memory
  cudaMemcpy (*d_qi, *h_qi, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_qe, *h_qe, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_mi, *h_mi, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_me, *h_me, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_kti, *h_kti, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_kte, *h_kte, sizeof(double), cudaMemcpyHostToDevice);

  // copy plasma properties from host to device memory
  cudaMemcpy (*d_n, *h_n, sizeof(double), cudaMemcpyHostToDevice);

  // copy probe properties from host to device memory
  cudaMemcpy (*d_phi_p, *h_phi_p, sizeof(double), cudaMemcpyHostToDevice);

  // copy geometrical properties from host to device memory
  cudaMemcpy (*d_Lx, *h_Lx, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_Ly, *h_Ly, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_dx, *h_dx, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_dy, *h_dy, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_dz, *h_dz, sizeof(double), cudaMemcpyHostToDevice);

  // copy electromagnetic properties from host to device memory
  cudaMemcpy (*d_epsilon, *h_epsilon, sizeof(double), cudaMemcpyHostToDevice);

  // copy mesh properties from host to device memory
  cudaMemcpy (*d_rho, *h_rho, ncx*ncy*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_phi, *h_phi, ncx*ncy*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_Ex, *h_Ex, ncx*ncy*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_Ey, *h_Ey, ncx*ncy*sizeof(double), cudaMemcpyHostToDevice);

  // copy timing variables from host to device memory
  cudaMemcpy (*d_t, *h_t, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_dt, *h_dt, sizeof(double), cudaMemcpyHostToDevice);

  // copy particle and bookmark vectors from host to device memory
  cudaMemcpy (*d_i, *h_i, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_e, *h_e, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_bookmarki, *h_bookmarki, (ncy-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_bookmarke, *h_bookmarke, (ncy-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);

  return;
}

/**********************************************************/

void read_input_file (double *h_qi, double *h_qe, double *h_mi, double *h_me, double *h_kti, double *h_kte, double *h_phi_p, double *h_n, double *h_Lx, double *h_Ly, double *h_dx, double *h_dy, double *h_dz, double *h_t, double *h_dt, double *h_epsilon)
{
  // function variables
  ifstream myfile;
  char line[80];

  // function body
  myfile.open("../input/input_data");
  if (myfile.is_open())
  {
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "q_i = %lf \n", h_qi);
    myfile.getline (line, 80);
    sscanf (line, "q_e = %lf \n", h_qe);
    myfile.getline (line, 80);
    sscanf (line, "m_i = %lf \n", h_mi);
    myfile.getline (line, 80);
    sscanf (line, "m_e = %lf \n", h_me);
    myfile.getline (line, 80);
    sscanf (line, "kT_i = %lf \n", h_kti);
    myfile.getline (line, 80);
    sscanf (line, "kT_e = %lf \n", h_kte);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "phi_p = %lf \n", h_phi_p);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "n = %lf \n", h_n);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "Lx = %lf \n", h_Lx);
    myfile.getline (line, 80);
    sscanf (line, "Ly = %lf \n", h_Ly);
    myfile.getline (line, 80);
    sscanf (line, "dx = %lf \n", h_dx);
    myfile.getline (line, 80);
    sscanf (line, "dy = %lf \n", h_dy);
    myfile.getline (line, 80);
    sscanf (line, "dz = %lf \n", h_dz);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "t = %lf \n", h_t);
    myfile.getline (line, 80);
    sscanf (line, "dt = %lf \n", h_dt);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "epsilon0 = %lf \n", h_epsilon);
  } else
  {
    cout << "input data file could not be opened" << endl;
    exit(1);
  }

  return;
}

/**********************************************************/

// fast particle-to-grid interpolation (based on the article by George Stantchev, William Dorland and Nail Gumerov)

void fast_particle_to_grid_interpolation (double dy, int ncy, int *bookmark, particle *p)
{
  // function variables

  // function body
//   particle_bining();
//   particle_to_cell_density_deposition();
//   cell_to_vertex_density_accumulation();

  return;
}

/**********************************************************/

void particle_bining(double dy, int ncy, int *bookmark, particle **p)
{
  // function variables
  int new_bookmark[2*ncy];
  dim3 griddim, blockdim;
  
  // function body
  
  // set dimensions of grid of threads for particle defragmentation
  griddim = ncy;
  blockdim = BINING_BLOCK_DIM;
  // execute kernel for defragmentation of particles
  particle_defragmentation<<<griddim, blockdim>>>(bookmark, new_bookmark, dy, *p);
  
  // set dimensions of grid of threads for particle rebracketing
  griddim = ncy-1;
  
  // execute kernel for rebracketing of particles
  particle_rebracketing<<<griddim, blockdim>>>(new_bookmark[0], bookmark, new_bookmark, *p);

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

__global__ void particle_rebracketing(int offset, int *bookmark, int *new_bookmark, particle *p)
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
    bookmark[1+blockIdx.x*2+threadIdx.x] = sh_new_bookmark[threadIdx.x]-offset;
  }
  __syncthreads();

  return;
}

/**********************************************************/
