/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "init.h"

/************************ FUNCTION DEFINITIONS ***********************/

void init_dev(void)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  int dev;
  int devcnt;
  cudaDeviceProp devProp;
  cudaError_t cuError;
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // check for devices instaled in the host
  cuError = cudaGetDeviceCount(&devcnt);
  if (0 != cuError)
  {
    printf("Cuda error (%d) detected in 'init_dev(void)'\n", cuError);
    cout << "exiting simulation..." << endl;
    exit(1);
  }
  cout << devcnt << " devices present in the host:" << endl;
  for (dev = 0; dev < devcnt; dev++) 
  {
    cudaGetDeviceProperties(&devProp, dev);
    cout << "  - Device " << dev << ":" << endl;
    cout << "    # " << devProp.name << endl;
    cout << "    # Compute capability " << devProp.major << "." << devProp.minor << endl;
  }

  // ask wich device to use
  cout << "Select in wich device simulation must be run: ";
  cin >> dev;
  
  // set device to be used and reset it
  cudaSetDevice(dev);
  cudaDeviceReset();
  
  return;
}

void init_sim(double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, int **d_e_bm, int **d_i_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double mi = init_mi();          // ion's mass
  const double me = init_me();          // electron's mass
  const double kti = init_kti();        // ion's thermal energy
  const double kte = init_kte();        // electron's thermal energy
  const double phi_p = init_phi_p();    // probe's potential
  const double n = init_n();            // plasma density
  const double Lx = init_Lx();          // size of the simulation in the x dimension (ccc)
  const double Ly = init_Ly();          // size of the simulation in the y dimension 
  const double ds = init_ds();          // spatial step size
  const double dt = init_dt();          // temporal step size
  const int ncy = init_ncy();           // number of cells in the y dimension
  const int nnx = init_nnx();           // number of nodes in the x dimension
  const int nny = init_nny();           // number of nodes in the y dimension
  
  int N;                                // initial number of particle of each species
  particle *h_i, *h_e;                  // host vectors of particles
  int *h_e_bm, *h_i_bm;                 // host vectors for bookmarks
  double *h_phi;                        // host vector for potentials
  
  dim3 griddim, blockdim;               // variables for kernel execution
  size_t sh_mem_size;
  cudaError_t cuError;
  
  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default); // default random number generator (gsl)
  
  // device memory
  double *d_Fx, *d_Fy;      // vectors for store the force that suffer each particle
  
  /*----------------------------- function body -------------------------*/
  
  // initialize enviromental variables for gsl random number generator
  gsl_rng_env_setup();
  
  // calculate initial number of particles
  N = int(n*Lx*ds*ds)*ncy;

  // allocate host memory for particle vectors
  h_i = (particle*) malloc(N*sizeof(particle));
  h_e = (particle*) malloc(N*sizeof(particle));

  // allocate host memory for bookmark vectors
  h_e_bm = (int*) malloc(2*ncy*sizeof(int));
  h_i_bm = (int*) malloc(2*ncy*sizeof(int));

  // allocate host memory for potential
  h_phi = (double*) malloc(nnx*nny*sizeof(double));

  // allocate device memory for particle vectors
  cuError = cudaMalloc (d_i, N*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc (d_e, N*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);

  // allocate device memory for bookmark vectors
  cuError = cudaMalloc (d_e_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc (d_i_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  
  // allocate device memory for mesh variables
  cuError = cudaMalloc (d_rho, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc (d_phi, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc (d_Ex, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc (d_Ey, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);

  // initialize particle vectors and bookmarks (host memory)
  for (int i = 0; i < ncy; i++)
  {
    h_e_bm[2*i] = i*(N/ncy);
    h_e_bm[2*i+1] = ((i+1)*(N/ncy))-1;
    h_i_bm[2*i] = i*(N/ncy);
    h_i_bm[2*i+1] = ((i+1)*(N/ncy))-1;
    for (int j = 0; j < N/ncy; j++)
    {
      // initialize ions
      h_i[(i*(N/ncy))+j].x = gsl_rng_uniform_pos(rng)*Lx;
      h_i[(i*(N/ncy))+j].y = double(i)*ds+gsl_rng_uniform_pos(rng)*ds;
      h_i[(i*(N/ncy))+j].vx = gsl_ran_gaussian(rng, sqrt(kti/mi));
      h_i[(i*(N/ncy))+j].vy = gsl_ran_gaussian(rng, sqrt(kti/mi));

      // initialize electrons
      h_e[(i*(N/ncy))+j].x = gsl_rng_uniform_pos(rng)*Lx;
      h_e[(i*(N/ncy))+j].y = double(i)*ds+gsl_rng_uniform_pos(rng)*ds;
      h_e[(i*(N/ncy))+j].vx = gsl_ran_gaussian(rng, sqrt(kte/me));
      h_e[(i*(N/ncy))+j].vy = gsl_ran_gaussian(rng, sqrt(kte/me));
    }
  }

  //initialize potential (host memory)
  for (int im = 0; im < nnx; im++)
  {
    for (int jm = 0; jm < nny; jm++)
    {
      h_phi[im+jm*(nnx)] = (1.0 - double(jm)/double(ncy))*phi_p;
    }
  }

  // copy particle and bookmark vectors from host to device memory
  cuError = cudaMemcpy (*d_i, h_i, N*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_e, h_e, N*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_i_bm, h_i_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_e_bm, h_e_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);

  // copy potential from host to device memory
  cuError = cudaMemcpy (*d_phi, h_phi, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // deposit charge into the mesh nodes
  charge_deposition((*d_rho), (*d_e), (*d_e_bm), (*d_i), (*d_i_bm));
  
  // solve poisson equation
  poisson_solver(1.0e-3, (*d_rho), (*d_phi));
  
  // derive electric fields from potential
  field_solver((*d_phi), (*d_Ex), (*d_Ey));
  
  // allocate device memory for particle forces
  cuError = cudaMalloc(&d_Fx, N*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc(&d_Fy, N*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // call kernels to calculate particle forces and fix their velocities
  griddim = ncy;     
  blockdim = PAR_MOV_BLOCK_DIM;
  sh_mem_size = 2*2*nnx*sizeof(double)+2*sizeof(int);
  
  // electrons (evaluate forces and fix velocities)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, -1, ds, (*d_e), (*d_e_bm), (*d_Ex), (*d_Ey), d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim>>>(dt, me, (*d_e), (*d_e_bm), d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // ions (evaluate forces and fix velocities)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, +1, ds, (*d_i), (*d_i_bm), (*d_Ex), (*d_Ey), d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim>>>(dt, mi, (*d_i), (*d_i_bm), d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // free device and host memories 
  free(h_i);
  free(h_e);
  free(h_i_bm);
  free(h_e_bm);
  free(h_phi);
  cuError = cudaFree(d_Fx);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_Fy);
  cu_check(cuError, __FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void read_input_file(double *ne, double *Te, double *beta, double *gamma, double *pot, int *ncx, int *ncy, double *ds, double *dt)
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
    sscanf (line, "ne = %lf \n", ne);
    myfile.getline (line, 80);
    sscanf (line, "Te = %lf \n", Te);
    myfile.getline (line, 80);
    sscanf (line, "beta = %lf \n", beta);
    myfile.getline (line, 80);
    sscanf (line, "gamma = %lf \n", gamma);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "phi_p = %lf \n", pot);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "ncx = %d \n", ncx);
    myfile.getline (line, 80);
    sscanf (line, "ncy = %d \n", ncy);
    myfile.getline (line, 80);
    sscanf (line, "ds = %lf \n", ds);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "dt = %lf \n", dt);
  } else
  {
    cout << "input data file could not be opened" << endl;
    exit(1);
  }

  return;
}

/**********************************************************/

double init_qi(void) 
{
  // function variables
  
  // function body
  
  return 1.0;
}

/**********************************************************/

double init_qe(void) 
{
  // function variables
  
  // function body
  
  return -1.0;
}

/**********************************************************/

double init_mi(void) 
{
  // function variables
  double ne, Te, beta, pot, ds, dt;
  int ncx, ncy;
  static double gamma = 0.0;

  // function body
  
  if (gamma == 0.0) read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
  
  return gamma;
}

/**********************************************************/

double init_me(void) 
{
  // function variables
  
  // function body
  
  return 1.0;
}

/**********************************************************/

double init_kti(void) 
{ 
  // function variables
  double ne, Te, gamma, pot, ds, dt;
  int ncx, ncy;
  static double beta = 0.0;
  
  // function body
  
  if (beta == 0.0) read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
  
  return beta;
}

/**********************************************************/

double init_kte(void) 
{
  // function variables
  
  // function body
  
  return 1.0;
}

/**********************************************************/

double init_phi_p(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  static double phi_p = 0.0;
  
  // function body
  
  if (phi_p == 0.0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    phi_p = pot*CST_E/(CST_KB*Te);
  }
  
  return phi_p;
}

/**********************************************************/

double init_n(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  const double Dl = init_Dl();
  static double n = 0.0;
  
  // function body
  
  if (n == 0.0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    n = ne*Dl*Dl*Dl;
  }
  
  return n;
}

/**********************************************************/

double init_Lx(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  static double Lx = 0.0;

  // function body
  
  if (Lx == 0.0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    Lx = double(ds*ncx);
  }
  
  return Lx;
}

/**********************************************************/

double init_Ly(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  static double Ly = 0.0;
  
  // function body
  
  if (Ly == 0.0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    Ly = double(ds*ncy);
  }
  
  return Ly;
}

/**********************************************************/

double init_ds(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, dt;
  int ncx, ncy;
  static double ds = 0.0;
  
  // function body
  
  if (ds == 0.0) read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
  
  return ds;
}

/**********************************************************/

double init_dt(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds;
  int ncx, ncy;
  static double dt = 0.0;
  
  // function body
  
  if (dt == 0.0) read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
  
  return dt;
}

/**********************************************************/

double init_epsilon0(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  const double Dl = init_Dl();
  static double epsilon0 = 0.0;
  // function body
  
  if (epsilon0 == 0.0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    epsilon0 = CST_EPSILON;
    epsilon0 /= pow(Dl*sqrt(CST_ME/(CST_KB*Te)),2); // time units
    epsilon0 /= CST_E*CST_E;                        // charge units
    epsilon0 *= Dl*Dl*Dl;                           // length units
    epsilon0 *= CST_ME;                             // mass units
  }
  
  return epsilon0;
}

/**********************************************************/

int init_ncx(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncy;
  static int ncx = 0;
  
  // function body
  
  if (ncx == 0) read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
  
  return ncx;
}

/**********************************************************/

int init_ncy(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx;
  static int ncy = 0;
  
  // function body
  
  if (ncy == 0) read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);

  return ncy;
}

/**********************************************************/

int init_nnx(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  static int nnx = 0;
  
  // function body
  
  if (nnx == 0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    nnx = ncx+1;
  }
  
  return nnx;
}

/**********************************************************/

int init_nny(void) 
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  static int nny = 0;
  
  // function body
  
  if (nny == 0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    nny = ncy+1;
  }
  
  return nny;
}

/**********************************************************/

double init_dtin_i(void)
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  const double n = init_n();
  const double Lx = init_Lx();
  static double dtin_i = 0;
  
  // function body
  
  if (dtin_i == 0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    dtin_i = sqrt(2.0*PI*gamma/beta)/(n*Lx*ds);
  }
  
  return dtin_i;
}

/**********************************************************/

double init_dtin_e(void)
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  const double n = init_n();
  const double Lx = init_Lx();
  static double dtin_e = 0;
  
  // function body
  
  if (dtin_e == 0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    dtin_e = sqrt(2.0*PI)/(n*Lx*ds);
  }
  
  return dtin_e;
}

/**********************************************************/

double init_Dl(void)
{
  // function variables
  double ne, Te, beta, gamma, pot, ds, dt;
  int ncx, ncy;
  static double Dl = 0;
  
  // function body
  
  if (Dl == 0) {
    read_input_file(&ne, &Te, &beta, &gamma, &pot, &ncx, &ncy, &ds, &dt);
    Dl = sqrt(CST_EPSILON*CST_KB*Te/(ne*CST_E*CST_E));
  }
  
  return Dl;
}

/**********************************************************/

/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void fix_velocity(double dt, double m, particle *g_p, int *g_bm, double *g_Fx, double *g_Fy) 
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
  if (threadIdx.x < 2)
  {
    sh_bm[threadIdx.x] = g_bm[blockIdx.x*2+threadIdx.x];
  }
  __syncthreads();
  
  //---- Process batches of particles
  
  for (int i = sh_bm[0]+threadIdx.x; i <= sh_bm[1]; i += blockDim.x)
  {
    // load particle data in registers
    p = g_p[i];
    Fx = g_Fx[i];
    Fy = g_Fy[i];
    
    // fix particle's velocity
    p.vx -= 0.5*dt*Fx/m;
    p.vy -= 0.5*dt*Fy/m;
    
    // store particle data in global memory
    g_p[i] = p;
  }
  
  return;
}

/**********************************************************/
