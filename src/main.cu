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
#include "diagnostic.h"

/************************ FUNCTION PROTOTIPES *************************/




/*************************** MAIN FUNCTION ****************************/

int main (int argc, const char* argv[])
{
  // host variables definition
  double t = 0;                         // time of simulation
  double dt = init_dt();                // time step

  // device variables definition
  double *d_rho, *d_phi, *d_Ex, *d_Ey;  // properties of mesh (charge density, potential and fields at nodes of the mesh)
  particle *d_e, *d_i;                  // vector of electron and ions
  int *d_e_bm, *d_i_bm;                 // vector that stores the bookmarks (beginning and end point) for each bin (electrons and ions)

  init_dev();
  init_sim(&d_rho, &d_phi, &d_Ex, &d_Ey, &d_e, &d_i, &d_e_bm, &d_i_bm);
  cout << "Simulation initialized with " << number_of_particles(d_e_bm)*2 << " particles." << endl << endl;
  cout << "t = " << 0. << endl;
  particles_snapshot(d_e, d_e_bm, "electrons_ic");
  particles_snapshot(d_i, d_i_bm, "ions_ic");
  
  for (int i = 1; i < 10000; i++, t += dt)
  {
    // deposit charge into the mesh nodes
    charge_deposition(d_rho, d_e, d_e_bm, d_i, d_i_bm);
    
    // solve poisson equation
    poisson_solver(1.0e-12, d_rho, d_phi);
    
    // derive electric fields from potential
    field_solver(d_phi, d_Ex, d_Ey);
    
    // move particles
    particle_mover(d_e, d_e_bm, d_i, d_i_bm, d_Ex, d_Ey);
    
    // contour condition
    cc(t, d_e_bm, &d_e, d_i_bm, &d_i, d_Ex, d_Ey);
    
    cout << "t = " << dt*i << endl;
  }
  
  cout << "Simulation finished!" << endl;
  return 0;
}
