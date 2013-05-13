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

void initialize (double **h_qi, double **h_qe, double **h_mi, double **h_me, double **h_kti, double **h_kte, double **h_phi_p, double **h_n, double **h_Lx, double **h_Ly, double **h_dx, double **h_dy, double **h_dz, double **h_t, double **h_dt, double **h_epsilon, double **h_rho, double **h_phi, double **h_Ex, double **h_Ey, particle **h_e, particle **h_i, unsigned int **h_bookmarke, unsigned int **h_bookmarki, double **d_qi, double **d_qe, double **d_mi, double **d_me, double **d_kti, double **d_kte, double **d_phi_p, double **d_n, double **d_Lx, double **d_Ly, double **d_dx, double **d_dy, double **d_dz, double **d_t, double **d_dt, double **d_epsilon, double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i, unsigned int **d_bookmarke, unsigned int **d_bookmarki)
{
  // function variables
  int N;                                          //initial number of particle of each species
  int ncx, ncy, nnx, nny;                         //number of cells and nodes in each dimension
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
  ncx = (**h_Lx)/(**h_dx);
  ncy = (**h_Ly)/(**h_dy);
  nnx = ncx + 1;
  nny = ncy + 1;
  N = int((**h_Lx)*(**h_dy)*(**h_dz)*(**h_n))*ncy;

  // allocate host memory for particle vectors
  *h_i = (particle*) malloc(N*sizeof(particle));
  *h_e = (particle*) malloc(N*sizeof(particle));

  // allocate host memory for bookmark vectors
  *h_bookmarke = (unsigned int*) malloc(2*ncy*sizeof(unsigned int));
  *h_bookmarki = (unsigned int*) malloc(2*ncy*sizeof(unsigned int));

  // allocate host memory for mesh variables
  *h_rho = (double*) malloc(nnx*nny*sizeof(double));
  *h_phi = (double*) malloc(nnx*nny*sizeof(double));
  *h_Ex = (double*) malloc(nnx*nny*sizeof(double));
  *h_Ey = (double*) malloc(nnx*nny*sizeof(double));

  // allocate device memory for particle vectors
  cudaMalloc (d_i, N*sizeof(particle));
  cudaMalloc (d_e, N*sizeof(particle));

  // allocate device memory for bookmark vectors
  cudaMalloc (d_bookmarke, 2*ncy*sizeof(unsigned int));
  cudaMalloc (d_bookmarki, 2*ncy*sizeof(unsigned int));

  // allocate device memory for mesh variables
  cudaMalloc (d_rho, nnx*nny*sizeof(double));
  cudaMalloc (d_phi, nnx*nny*sizeof(double));
  cudaMalloc (d_Ex, nnx*nny*sizeof(double));
  cudaMalloc (d_Ey, nnx*nny*sizeof(double));

  // initialize particle vectors and bookmarks (host memory)
  for (int i = 0; i < ncy-1; i++)
  {
    (*h_bookmarke)[2*i] = i*N/ncy;
    (*h_bookmarke)[2*i+1] = ((i+1)*N/ncy)-1;
    (*h_bookmarki)[2*i] = i*N/ncy;
    (*h_bookmarki)[2*i+1] = ((i+1)*N/ncy)-1;
    for (int j = 0; j < N/ncy; j++)
    {
      // initialize ions
      (*h_i)[(i*N/ncy)+j].x = gsl_rng_uniform_pos(rng)*(**h_Lx);
      (*h_i)[(i*N/ncy)+j].y = double(i)*(**h_dy)+gsl_rng_uniform_pos(rng)*(**h_dy);
      (*h_i)[(i*N/ncy)+j].vx = gsl_ran_gaussian(rng, sqrt((**h_kti)/(**h_mi)));
      (*h_i)[(i*N/ncy)+j].vy = gsl_ran_gaussian(rng, sqrt((**h_kti)/(**h_mi)));

      // initialize electrons
      (*h_e)[(i*N/ncy)+j].x = gsl_rng_uniform_pos(rng)*(**h_Lx);
      (*h_e)[(i*N/ncy)+j].y = double(i)*(**h_dy)+gsl_rng_uniform_pos(rng)*(**h_dy);
      (*h_e)[(i*N/ncy)+j].vx = gsl_ran_gaussian(rng, sqrt((**h_kte)/(**h_me)));
      (*h_e)[(i*N/ncy)+j].vy = gsl_ran_gaussian(rng, sqrt((**h_kte)/(**h_me)));
    }
  }

  //initialize mesh variables (host memory)
  for (int im = 0; im < nnx; im++)
  {
    for (int jm = 0; jm < nny; jm++)
    {
      (*h_Ex)[im+jm*(nnx)] = 0.0;
      (*h_Ey)[im+jm*(nnx)] = 0.0;
      (*h_rho)[im+jm*(nnx)] = 0.0;
      (*h_phi)[im+jm*(nnx)] = (1.0 - double(jm)/double(ncy))*(**h_phi_p);
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
  cudaMemcpy (*d_rho, *h_rho, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_phi, *h_phi, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_Ex, *h_Ex, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_Ey, *h_Ey, nnx*nny*sizeof(double), cudaMemcpyHostToDevice);

  // copy timing variables from host to device memory
  cudaMemcpy (*d_t, *h_t, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_dt, *h_dt, sizeof(double), cudaMemcpyHostToDevice);

  // copy particle and bookmark vectors from host to device memory
  cudaMemcpy (*d_i, *h_i, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_e, *h_e, N*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_bookmarki, *h_bookmarki, 2*(ncy-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_bookmarke, *h_bookmarke, 2*(ncy-1)*sizeof(unsigned int), cudaMemcpyHostToDevice);

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