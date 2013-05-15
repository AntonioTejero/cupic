/****************************************************************************
 *                                                                          *
 *    CUPIC is a code that simulates the interaction between plasma and     *
 *    a langmuir probe using PIC techniques accelerated with the use of     *
 *    GPU hardware (CUDA extension of C/C++)                                *
 *                                                                          *
 ****************************************************************************/


/****************************** HEADERS ******************************/

#include "stdh.h"
#include "init.h"
#include "cc.h"
#include "mesh.h"
#include "particles.h"

/************************ FUNCTION PROTOTIPES *************************/

void fast_particle_to_grid_interpolation (double dy, int ncy, int *bookmark, particle **p);
void particle_bining(double dy, int ncy, int *bookmark, particle **p);


/*************************** MAIN FUNCTION ****************************/

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