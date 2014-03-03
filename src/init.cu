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
  cout << "Select in wich device simulation must be run: 0" << endl;
  dev = 0;  //cin >> dev;
  
  // set device to be used and reset it
  cudaSetDevice(dev);
  cudaDeviceReset();
  
  return;
}

void init_sim(double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, int **d_e_bm, int **d_i_bm, double *t)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double dt = init_dt();
  const int n_ini = init_n_ini();

  // device memory
  
  /*----------------------------- function body -------------------------*/

  // check if simulation start from initial condition or saved state
  if (n_ini == 0) {
    // adjust initial time
    *t = 0.;

    // create particles
    create_particles(d_i, d_i_bm, d_e, d_e_bm);

    // initialize mesh variables
    initialize_mesh(d_rho, d_phi, d_Ex, d_Ey, *d_i, *d_i_bm, *d_e, *d_e_bm);

    // adjust velocities for leap-frog scheme
    adjust_leap_frog(*d_i, *d_i_bm, *d_e, *d_e_bm, *d_Ex, *d_Ey);
    
    cout << "Simulation initialized with " << number_of_particles(*d_e_bm)*2 << " particles." << endl << endl;
  } else if (n_ini > 0) {
    // adjust initial time
    *t = n_ini*dt;

    // read particle from file
    load_particles(d_i, d_i_bm, d_e, d_e_bm);
    
    // initialize mesh variables
    initialize_mesh(d_rho, d_phi, d_Ex, d_Ey, *d_i, *d_i_bm, *d_e, *d_e_bm);

    cout << "Simulation state loaded from time t = " << *t << endl;
  } else {
    cout << "Wrong input parameter (n_ini<0)" << endl;
    cout << "Stoppin simulation" << endl;
    exit(1);
  }
  
  return;
}

/**********************************************************/

void create_particles(particle **d_i, int **d_i_bm, particle **d_e, int **d_e_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double n = init_n();        // plasma density
  const double mi = init_mi();      // ion's mass
  const double me = init_me();      // electron's mass
  const double kti = init_kti();    // ion's thermal energy
  const double kte = init_kte();    // electron's thermal energy
  const double Lx = init_Lx();      // size of the simulation in the x dimension (ccc)
  const double ds = init_ds();      // spatial step size
  const int ncy = init_ncy();       // number of cells in the y dimension
  
  particle *h_i, *h_e;              // host vectors of particles
  int *h_e_bm, *h_i_bm;             // host vectors for bookmarks
  int N;                            // initial number of particles of each especie

  gsl_rng *rng = gsl_rng_alloc(gsl_rng_default); // default random number generator (gsl)

  cudaError_t cuError;              // cuda error variable
  
  // device memory

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

  // allocate device memory for particle vectors
  cuError = cudaMalloc ((void **) (void **) d_i, N*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_e, N*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  
  // allocate device memory for bookmark vectors
  cuError = cudaMalloc ((void **) d_e_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_i_bm, 2*ncy*sizeof(int));
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

  // copy particle and bookmark vectors from host to device memory
  cuError = cudaMemcpy (*d_i, h_i, N*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_e, h_e, N*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_i_bm, h_i_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_e_bm, h_e_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);

  // free host memory
  free(h_i);
  free(h_e);
  free(h_i_bm);
  free(h_e_bm);
  
  return;
}

/**********************************************************/

void initialize_mesh(double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle *d_i, int *d_i_bm, particle *d_e, int *d_e_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double phi_p = init_phi_p();    // probe's potential
  const int nnx = init_nnx();           // number of nodes in the x dimension
  const int nny = init_nny();           // number of nodes in the y dimension
  const int ncy = init_ncy();           // number of cells in the y dimension
  

  double *h_phi;                        // host vector for potentials
  
  cudaError_t cuError;                  // cuda error variable
  
  // device memory
  
  /*----------------------------- function body -------------------------*/
  
  // allocate host memory for potential
  h_phi = (double*) malloc(nnx*nny*sizeof(double));
  
  // allocate device memory for mesh variables
  cuError = cudaMalloc ((void **) d_rho, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_phi, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_Ex, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_Ey, nnx*nny*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  //initialize potential (host memory)
  for (int im = 0; im < nnx; im++)
  {
    for (int jm = 0; jm < nny; jm++)
    {
      h_phi[im+jm*(nnx)] = (1.0 - double(jm)/double(ncy))*phi_p;
    }
  }
  
  // copy potential from host to device memory
  cuError = cudaMemcpy (*d_phi, h_phi, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  
  // free host memory
  free(h_phi);
  
  // deposit charge into the mesh nodes
  charge_deposition((*d_rho), d_e, d_e_bm, d_i, d_i_bm);
  
  // solve poisson equation
  poisson_solver(1.0e-3, (*d_rho), (*d_phi));
  
  // derive electric fields from potential
  field_solver((*d_phi), (*d_Ex), (*d_Ey));
  
  return;
}

/**********************************************************/

void adjust_leap_frog(particle *d_i, int *d_i_bm, particle *d_e, int *d_e_bm, double *d_Ex, double *d_Ey)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double mi = init_mi();          // ion's mass
  const double me = init_me();          // electron's mass
  const double ds = init_ds();          // spatial step size
  const double dt = init_dt();          // temporal step size
  const int ncy = init_ncy();           // number of cells in the y dimension
  const int nnx = init_nnx();           // number of nodes in the x dimension
  
  int N = number_of_particles(d_i_bm);  // number of particles of each especie (same for both)

  dim3 griddim, blockdim;               // kernel execution configurations
  size_t sh_mem_size;                   // shared memory size
  cudaError_t cuError;                  // cuda error variable
  
  // device memory
  double *d_Fx, *d_Fy;      // vectors for store the force that suffer each particle
  
  /*----------------------------- function body -------------------------*/

  // allocate device memory for particle forces
  cuError = cudaMalloc((void **) &d_Fx, N*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc((void **) &d_Fy, N*sizeof(double));
  cu_check(cuError, __FILE__, __LINE__);
  
  // call kernels to calculate particle forces and fix their velocities
  griddim = ncy;
  blockdim = PAR_MOV_BLOCK_DIM;
  sh_mem_size = 2*2*nnx*sizeof(double)+2*sizeof(int);
  
  // electrons (evaluate forces and fix velocities)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, -1, ds, d_e, d_e_bm, d_Ex, d_Ey, d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim>>>(dt, me, d_e, d_e_bm, d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // ions (evaluate forces and fix velocities)
  cudaGetLastError();
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, +1, ds, d_i, d_i_bm, d_Ex, d_Ey, d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  cudaGetLastError();
  fix_velocity<<<griddim, blockdim>>>(dt, mi, d_i, d_i_bm, d_Fx, d_Fy);
  cu_sync_check(__FILE__, __LINE__);
  
  // free device and host memory
  cuError = cudaFree(d_Fx);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaFree(d_Fy);
  cu_check(cuError, __FILE__, __LINE__);
  
  return;
}

/**********************************************************/

void load_particles(particle **d_i, int **d_i_bm, particle **d_e, int **d_e_bm)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  char filename[50];

  // device memory

  /*----------------------------- function body -------------------------*/

  sprintf(filename, "./ions.dat");
  read_particle_file(filename, d_i, d_i_bm);
  sprintf(filename, "./electrons.dat");
  read_particle_file(filename, d_e, d_e_bm);
  
  return;
}

/**********************************************************/

void read_particle_file(string filename, particle **d_p, int **d_bm)
{
  /*--------------------------- function variables -----------------------*/

  // host memory
  const int ncy = init_ncy();   // number of cells in the y dimension
  const double ds = init_ds();  // space step
  particle *h_p;                // host vector for particles
  int *h_bm;                    // host vector for bookmarks
  int n = 0;                    // number of particles
  int bin;                      // bin
  
  ifstream myfile;              // file variables
  char line[150];

  cudaError_t cuError;          // cuda error variable
  
  // device memory

  /*----------------------------- function body -------------------------*/

  // get number of particles
  myfile.open(filename.c_str());
  if (myfile.is_open()) {
    myfile.getline(line, 150);
    while (!myfile.eof()) {
      myfile.getline(line, 150);
      n++;
    }
    n--;
  }
  myfile.close();

  // allocate host and device memory for particles and bookmarks
  h_p = (particle*) malloc(n*sizeof(particle));
  h_bm = (int*) malloc(2*ncy*sizeof(int));
  cuError = cudaMalloc ((void **) d_p, n*sizeof(particle));
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMalloc ((void **) d_bm, 2*ncy*sizeof(int));
  cu_check(cuError, __FILE__, __LINE__);
  
  // read particles from file and store in host memory
  myfile.open(filename.c_str());
  if (myfile.is_open()) {
    myfile.getline(line, 150);
    for (int i = 0; i<n; i++) {
      myfile.getline(line, 150);
      sscanf (line, " %le %le %le %le \n", &h_p[i].x, &h_p[i].y, &h_p[i].vx, &h_p[i].vy);
    }
  }
  myfile.close();

  // calculate bookmarks and store in host memory
  for (int i = 0; i < 2*ncy; i++) h_bm[i]=-1;
  for (int i = 0; i < n; ) {
    bin = int(h_p[i].y/ds);
    h_bm[bin*2] = i;
    while (bin == int(h_p[i].y/ds) && i < n) i++;
    h_bm[bin*2+1] = i-1;
  }

  // copy particle and bookmark vector from host to device memory
  cuError = cudaMemcpy (*d_p, h_p, n*sizeof(particle), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);
  cuError = cudaMemcpy (*d_bm, h_bm, 2*ncy*sizeof(int), cudaMemcpyHostToDevice);
  cu_check(cuError, __FILE__, __LINE__);

  // free host memory
  free(h_p);
  free(h_bm);
  
  return;
}

/**********************************************************/

void read_input_file(void *data, int data_size, int n)
{
  // function variables
  ifstream myfile;
  char line[80];

  // function body
  myfile.open("../input/input_data");
  if (myfile.is_open()) {
    myfile.getline(line, 80);
    for (int i = 0; i < n; i++) myfile.getline(line, 80);
    if (data_size == sizeof(int)) {
      sscanf (line, "%*s = %d;\n", (int*) data);
    } else if (data_size == sizeof(double)) {
      sscanf (line, "%*s = %lf;\n", (double*) data);
    }
  } else {
    cout << "input data file could not be opened" << endl;
    exit(1);
  }
  myfile.close();
  
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
  static double gamma = 0.0;

  // function body
  
  if (gamma == 0.0) read_input_file((void*) &gamma, sizeof(gamma), 8);
  
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
  static double beta = 0.0;
  
  // function body
  
  if (beta == 0.0) read_input_file((void*) &beta, sizeof(beta), 7);
  
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
  double Te;
  static double phi_p = 0.0;
  
  // function body
  
  if (phi_p == 0.0) {
    read_input_file((void*) &Te, sizeof(Te), 6);
    read_input_file((void*) &phi_p, sizeof(phi_p), 9);
    phi_p *= CST_E/(CST_KB*Te);
  }
  
  return phi_p;
}

/**********************************************************/

double init_n(void) 
{
  // function variables
  const double Dl = init_Dl();
  static double n = 0.0;
  
  // function body
  
  if (n == 0.0) {
    read_input_file((void*) &n, sizeof(n), 5);
    n *= Dl*Dl*Dl;
  }
  
  return n;
}

/**********************************************************/

double init_Lx(void) 
{
  // function variables
  static double Lx = double(init_ds()*init_ncx());

  // function body
  
  return Lx;
}

/**********************************************************/

double init_Ly(void) 
{
  // function variables
  static double Ly = double(init_ds()*init_ncy());
  
  // function body
  
  return Ly;
}

/**********************************************************/

double init_ds(void) 
{
  // function variables
  static double ds = 0.0;
  
  // function body
  
  if (ds == 0.0) read_input_file((void*) &ds, sizeof(double), 12);
  
  return ds;
}

/**********************************************************/

double init_dt(void) 
{
  // function variables
  static double dt = 0.0;
  
  // function body
  
  if (dt == 0.0) read_input_file((void*) &dt, sizeof(double), 13);
  
  return dt;
}

/**********************************************************/

double init_epsilon0(void) 
{
  // function variables
  double Te;
  const double Dl = init_Dl();
  static double epsilon0 = 0.0;
  // function body
  
  if (epsilon0 == 0.0) {
    read_input_file((void*) &Te, sizeof(Te), 6);
    epsilon0 = CST_EPSILON;                         // SI units
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
  static int ncx = 0;
  
  // function body
  
  if (ncx == 0) read_input_file((void*) &ncx, sizeof(ncx), 10);
  
  return ncx;
}

/**********************************************************/

int init_ncy(void) 
{
  // function variables
  static int ncy = 0;
  
  // function body
  
  if (ncy == 0) read_input_file((void*) &ncy, sizeof(ncy), 11);

  return ncy;
}

/**********************************************************/

int init_nnx(void) 
{
  // function variables
  static int nnx = init_ncx()+1;
  
  // function body
  
  return nnx;
}

/**********************************************************/

int init_nny(void) 
{
  // function variables
  static int nny = init_ncy()+1;
  
  // function body
  
  return nny;
}

/**********************************************************/

double init_dtin_i(void)
{
  // function variables
  const double mi = init_mi();
  const double kti = init_kti();
  const double n = init_n();
  const double Lx = init_Lx();
  const double ds = init_ds();
  static double dtin_i = sqrt(2.0*PI*mi/kti)/(n*Lx*ds);
  
  // function body
  
  return dtin_i;
}

/**********************************************************/

double init_dtin_e(void)
{
  // function variables
  const double n = init_n();
  const double Lx = init_Lx();
  const double ds = init_ds();
  static double dtin_e = sqrt(2.0*PI)/(n*Lx*ds);
  
  // function body
  
  return dtin_e;
}

/**********************************************************/

double init_Dl(void)
{
  // function variables
  double ne, Te;
  static double Dl = 0.0;
  
  // function body
  
  if (Dl == 0.0) {
    read_input_file((void*) &ne, sizeof(ne), 5);
    read_input_file((void*) &Te, sizeof(Te), 6);
    Dl = sqrt(CST_EPSILON*CST_KB*Te/(ne*CST_E*CST_E));
  }
  
  return Dl;
}

/**********************************************************/

int init_n_ini(void)
{
  // function variables
  static int n_ini = -1;
  
  // function body
  
  if (n_ini < 0) read_input_file((void*) &n_ini, sizeof(n_ini), 1);
  
  return n_ini;
}

/**********************************************************/

int init_n_prev(void)
{
  // function variables
  static int n_prev = -1;
  
  // function body
  
  if (n_prev < 0) read_input_file((void*) &n_prev, sizeof(n_prev), 2);
  
  return n_prev;
}

/**********************************************************/

int init_n_save(void)
{
  // function variables
  static int n_save = -1;
  
  // function body
  
  if (n_save < 0) read_input_file((void*) &n_save, sizeof(n_save), 3);
  
  return n_save;
}

/**********************************************************/

int init_n_fin(void)
{
  // function variables
  static int n_fin = -1;
  
  // function body
  
  if (n_fin < 0) read_input_file((void*) &n_fin, sizeof(n_fin), 4);
  
  return n_fin;
}

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
