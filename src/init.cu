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

void initialize (double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, unsigned int **d_e_bm, unsigned int **d_i_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  const double qi = init_qi();                // ion's charge
  const double qe = init_qe();                // electron's charge
  const double mi = init_mi();                // ion's mass
  const double me = init_me();                // electron's mass
  const double kti = init_kti();              // ion's thermal energy
  const double kte = init_kte();              // electron's thermal energy
  const double phi_p = init_phi_p();          // probe's potential
  const double n = init_n();                  // plasma density
  const double Lx = init_Lx();                // size of the simulation in the x dimension (ccc)
  const double Ly = init_Ly();                // size of the simulation in the y dimension 
  const double ds = init_ds();                // spatial step size
  const double dt = init_dt();                // temporal step size
  const double epsilon0 = init_epsilon0();    // permittivity of free space
  const int ncx = init_ncx();                 // number of cells in the x dimension
  const int ncy = init_ncy();                 // number of cells in the y dimension
  const int nnx = init_nnx();                 // number of nodes in the x dimension
  const int nny = init_nny();                 // number of nodes in the y dimension
  
  int N;                                      // initial number of particle of each species
  particle *h_i, *h_e;                        // host vectors of particles
  unsigned int *h_e_bm, *h_i_bm;              // host vectors for bookmarks
  double *h_phi;                              // host vector for potentials
  
  dim3 griddim, blockdim;
  size_t sh_mem_size;
  
  gsl_rng * rng = gsl_rng_alloc(gsl_rng_default); // default random number generator (gsl)
  
  // device memory
  double *d_Fx, *d_Fy;                        // vectors for store the force that suffer each particle
  
  /*----------------------------- function body -------------------------*/
  
  // initialize enviromental variables for gsl random number generator
  gsl_rng_env_setup();
  
  // calculate initial number of particles
  N = int(Lx*ds*ds)*ncy;

  // allocate host memory for particle vectors
  h_i = (particle*) malloc(N*sizeof(particle));
  h_e = (particle*) malloc(N*sizeof(particle));

  // allocate host memory for bookmark vectors
  h_e_bm = (unsigned int*) malloc(2*ncy*sizeof(unsigned int));
  h_i_bm = (unsigned int*) malloc(2*ncy*sizeof(unsigned int));

  // allocate host memory for potential
  h_phi = (double*) malloc(nnx*nny*sizeof(double));

  // allocate device memory for particle vectors
  cudaMalloc (d_i, N*sizeof(particle));
  cudaMalloc (d_e, N*sizeof(particle));

  // allocate device memory for bookmark vectors
  cudaMalloc (d_e_bm, 2*ncy*sizeof(unsigned int));
  cudaMalloc (d_i_bm, 2*ncy*sizeof(unsigned int));

  // allocate device memory for mesh variables
  cudaMalloc (d_rho, nnx*nny*sizeof(double));
  cudaMalloc (d_phi, nnx*nny*sizeof(double));
  cudaMalloc (d_Ex, nnx*nny*sizeof(double));
  cudaMalloc (d_Ey, nnx*nny*sizeof(double));

  // initialize particle vectors and bookmarks (host memory)
  for (int i = 0; i < ncy-1; i++)
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
  cudaMemcpy (*d_i, h_i, N*sizeof(particle), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_e, h_e, N*sizeof(particle), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_i_bm, h_i_bm, 2*(ncy-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_e_bm, h_e_bm, 2*(ncy-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);

  // copy potential from host to device memory
  cudaMemcpy (*d_phi, h_phi, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  
  // deposit charge into the mesh nodes
  charge_deposition((*d_rho), (*d_e), (*d_e_bm), (*d_i), (*d_i_bm));
  
  // solve poisson equation
  poisson_solver(1.0e-12, (*d_rho), (*d_phi));
  
  // derive electric fields from potential
  field_solver((*d_phi), (*d_Ex), (*d_Ey));
  
  // allocate device memory for particle forces
  cudaMalloc(&d_Fx, N*sizeof(double));
  cudaMalloc(&d_Fy, N*sizeof(double));
  
  // call kernels to calculate particle forces and fix their velocities
  griddim = ncy;                                                  // set dimensions of grid of blocks for fast_grid_to_particle and fix_velocity kernel
  blockdim = PAR_MOV_BLOCK_DIM;                                   // set dimensions of block of threads for fast_grid_to_particle and fix_velocity kernel
  sh_mem_size = 2*2*nnx*sizeof(double)+2*sizeof(unsigned int);    // define size of shared memory for fast_grid_to_particle kernel
  
  // electrons
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, -1, ds, (*d_e), (*d_e_bm), (*d_Ex), (*d_Ey), d_Fx, d_Fy);    // calculate forces
  fix_velocity<<<griddim, blockdim>>>(dt, me, (*d_e), (*d_e_bm), d_Fx, d_Fy);                                                 // fix electron's velocities
  // ions
  fast_grid_to_particle<<<griddim, blockdim, sh_mem_size>>>(nnx, +1, ds, (*d_i), (*d_i_bm), (*d_Ex), (*d_Ey), d_Fx, d_Fy);    // calculate forces
  fix_velocity<<<griddim, blockdim>>>(dt, mi, (*d_i), (*d_i_bm), d_Fx, d_Fy);                                                 // fix ion's velocities
  
  // free device and host memories 
  free(h_i);
  free(h_e);
  free(h_i_bm);
  free(h_e_bm);
  free(h_phi);
  cudaFree(d_Fx);
  cudaFree(d_Fy);
  
  
  return;
}

/**********************************************************/

void read_input_file (double *qi, double *qe, double *mi, double *me, double *kti, double *kte, double *phi_p, double *n, double *Lx, double *Ly, double *ds, double *dt, double *epsilon0)
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
    sscanf (line, "q_i = %lf \n", qi);
    myfile.getline (line, 80);
    sscanf (line, "q_e = %lf \n", qe);
    myfile.getline (line, 80);
    sscanf (line, "m_i = %lf \n", mi);
    myfile.getline (line, 80);
    sscanf (line, "m_e = %lf \n", me);
    myfile.getline (line, 80);
    sscanf (line, "kT_i = %lf \n", kti);
    myfile.getline (line, 80);
    sscanf (line, "kT_e = %lf \n", kte);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "phi_p = %lf \n", phi_p);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "n = %lf \n", n);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "Lx = %lf \n", Lx);
    myfile.getline (line, 80);
    sscanf (line, "Ly = %lf \n", Ly);
    myfile.getline (line, 80);
    sscanf (line, "dx = %lf \n", ds);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "dt = %lf \n", dt);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    myfile.getline (line, 80);
    sscanf (line, "epsilon0 = %lf \n", epsilon0);
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
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return qi;
}

/**********************************************************/

double init_qe(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return qe;
}

/**********************************************************/

double init_mi(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return mi;
}

/**********************************************************/

double init_me(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return me;
}

/**********************************************************/

double init_kti(void) 
{ 
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return kti;
}

/**********************************************************/

double init_kte(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return kte;
}

/**********************************************************/

double init_phi_p(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return phi_p;
}

/**********************************************************/

double init_n(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return n;
}

/**********************************************************/

double init_Lx(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return Lx;
}

/**********************************************************/

double init_Ly(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return Ly;
}

/**********************************************************/

double init_ds(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return ds;
}

/**********************************************************/

double init_dt(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return dt;
}

/**********************************************************/

double init_epsilon0(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  
  return epsilon0;
}

/**********************************************************/

int init_ncx(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  int ncx;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  ncx = int(Lx/ds);
  
  return ncx;
}

/**********************************************************/

int init_ncy(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  int ncy;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  ncy = int(Ly/ds);
  
  return ncy;
}

/**********************************************************/

int init_nnx(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  int nnx;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  nnx = int(Lx/ds)+1;
  
  return nnx;
}

/**********************************************************/

int init_nny(void) 
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  int nny;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  nny = int(Ly/ds)+1;
  
  return nny;
}

/**********************************************************/

double init_dtin_i(void)
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  double dtin_i;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  dtin_i = sqrt(2.0*PI*mi/kti)/(n*Lx*ds);
  
  return dtin_i;
}

/**********************************************************/

double init_dtin_e(void)
{
  // function variables
  double qi, qe, mi, me, kti, kte, phi_p, n, Lx, Ly, ds, dt, epsilon0;
  double dtin_e;
  
  // function body
  
  read_input_file(&qi, &qe, &mi, &me, &kti, &kte, &phi_p, &n, &Lx, &Ly, &ds, &dt, &epsilon0);
  dtin_e = sqrt(2.0*PI*me/kte)/(n*Lx*ds);
  
  return dtin_e;
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/

__global__ void fix_velocity(double dt, double m, particle *g_p, unsigned int *g_bm, double *g_Fx, double *g_Fy) 
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
    
    // fix particle's velocity
    p.vx -= 0.5*dt*Fx/m;
    p.vy -= 0.5*dt*Fy/m;
    
    // store particle data in global memory
    g_p[i] = p;
  }
  
  return;
}

/**********************************************************/