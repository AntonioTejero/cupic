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
#include <gsl/gsl_rng.h>	  		//gsl library for random number generation
#include <gsl/gsl_randist.h>		//gsl library for random number generation

using namespace std;

#define PI 3.1415926535897932		//symbolic constant for PI

struct particle
{
  double x;
  double y;
  double vx;
  double vy;
};

/****************************** FUNCTION PROTOTIPES ******************************/

void initialize (double **h_qi, double **h_qe, double **h_mi, double **h_me, double **h_kti, double **h_kte, double **h_phi_p, double **h_n, double **h_Lx, double **h_Ly, double **h_dx, double **h_dy, double **h_dz, double **h_t, double **h_dt, double **h_epsilon, double **h_rho, double **h_phi, double **h_Ex, double **h_Ey, particle **h_e, particle **h_i, double **d_qi, double **d_qe, double **d_mi, double **d_me, double **d_kti, double **d_kte, double **d_phi_p, double **d_n, double **d_Lx, double **d_Ly, double **d_dx, double **d_dy, double **d_dz, double **d_t, double **d_dt, double **d_epsilon, double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i);

void read_input_file (double *h_qi, double *h_qe, double *h_mi, double *h_me, double *h_kti, double *h_kte, double *h_phi_p, double *h_n, double *h_Lx, double *h_Ly, double *h_dx, double *h_dy, double *h_dz, double *h_t, double *h_dt, double *h_epsilon);

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

  // device variables definition
  double *d_qi, *d_qe, *d_mi, *d_me, *d_kti, *d_kte;  //properties of particles (charge, mass and temperature of particle species)
  double *d_n;                                        //plasma properties (plasma density)
  double *d_phi_p;                                    //probe properties (probe potential)
  double *d_Lx, *d_Ly, *d_dx, *d_dy, *d_dz;           //geometrical properties of simulation (simulation dimensions and spacial step)
  double *d_epsilon;                                  //electromagnetic properties
  double *d_rho, *d_phi, *d_Ex, *d_Ey;                //properties of mesh (charge density, potential and fields at point of the mesh)
  double *d_t, *d_dt;                                 //timing variables (simulation time and time step)
  particle *d_e, *d_i;                                //vector of electron and ions


  initialize (&h_qi, &h_qe, &h_mi, &h_me, &h_kti, &h_kte, &h_phi_p, &h_n, &h_Lx, &h_Ly, &h_dx, &h_dy, &h_dz, &h_t, &h_dt, &h_epsilon, &h_rho, &h_phi, &h_Ex, &h_Ey, &h_e, &h_i, &d_qi, &d_qe, &d_mi, &d_me, &d_kti, &d_kte, &d_phi_p, &d_n, &d_Lx, &d_Ly, &d_dx, &d_dy, &d_dz, &d_t, &d_dt, &d_epsilon, &d_rho, &d_phi, &d_Ex, &d_Ey, &d_e, &d_i);

  return 0;
}

/****************************** FUNCTION DEFINITION ******************************/

void initialize (double **h_qi, double **h_qe, double **h_mi, double **h_me, double **h_kti, double **h_kte, double **h_phi_p, double **h_n, double **h_Lx, double **h_Ly, double **h_dx, double **h_dy, double **h_dz, double **h_t, double **h_dt, double **h_epsilon, double **h_rho, double **h_phi, double **h_Ex, double **h_Ey, particle **h_e, particle **h_i, double **d_qi, double **d_qe, double **d_mi, double **d_me, double **d_kti, double **d_kte, double **d_phi_p, double **d_n, double **d_Lx, double **d_Ly, double **d_dx, double **d_dy, double **d_dz, double **d_t, double **d_dt, double **d_epsilon, double **d_rho, double **d_phi, double **d_Ex, double **d_Ey, particle **d_e, particle **d_i)
{
  // allocate host memory for particle properties
  *h_qi = (double*) malloc(sizeof(double));
  *h_qe = (double*) malloc(sizeof(double));
  *h_mi = (double*) malloc(sizeof(double));
  *h_me = (double*) malloc(sizeof(double));
  *h_kti = (double*) malloc(sizeof(double));
  *h_kte = (double*) malloc(sizeof(double));

  // read input file
  read_input_file (*h_qi, *h_qe, *h_mi, *h_me, *h_kti, *h_kte, *h_phi_p, *h_n, *h_Lx, *h_Ly, *h_dx, *h_dy, *h_dz, *h_t, *h_dt, *h_epsilon);

  // allocate device memory for particle properties
  cudaMalloc (d_qi, sizeof(double));
  cudaMalloc (d_qe, sizeof(double));
  cudaMalloc (d_mi, sizeof(double));
  cudaMalloc (d_me, sizeof(double));
  cudaMalloc (d_kti, sizeof(double));
  cudaMalloc (d_kte, sizeof(double));

  // copy particle properties from host to device memory
  cudaMemcpy (*d_qi, *h_qi, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_qe, *h_qe, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_mi, *h_mi, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_me, *h_me, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_kti, *h_kti, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy (*d_kte, *h_kte, sizeof(double), cudaMemcpyHostToDevice);


  return;
}

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
  }

  return;
}
