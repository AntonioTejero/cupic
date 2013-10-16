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
  double t = 0;                       // time of simulation
  const double dt = init_dt();        // time step
  const int n_prev = init_n_prev();   // number of iterations before start analizing
  const int n_save = init_n_save();   // number of iterations between diagnostics
  const int n_total = init_n_total(); // number of total iterations
  char filename[50];                  // filename for saved data

  // device variables definition
  double *d_rho, *d_phi, *d_Ex, *d_Ey;  // properties of mesh (charge density, potential and fields at nodes of the mesh)
  particle *d_e, *d_i;                  // vector of electron and ions
  int *d_e_bm, *d_i_bm;                 // vector that stores the bookmarks (beginning and end point) for each bin (electrons and ions)

  init_dev();
  init_sim(&d_rho, &d_phi, &d_Ex, &d_Ey, &d_e, &d_i, &d_e_bm, &d_i_bm);
  cout << "Simulation initialized with " << number_of_particles(d_e_bm)*2 << " particles." << endl << endl;

  sprintf(filename, "e_t_0");
  particles_snapshot(d_e, d_e_bm, filename);
  sprintf(filename, "i_t_0");
  particles_snapshot(d_i, d_i_bm, filename);
  sprintf(filename, "charge_t_0");
  mesh_snapshot(d_rho, filename);
  sprintf(filename, "potential_t_0");
  mesh_snapshot(d_phi, filename);

  for (int i = 0; i < n_total; i++, t += dt) {
    cout << "t = " << t << endl;
    
    // deposit charge into the mesh nodes
    charge_deposition(d_rho, d_e, d_e_bm, d_i, d_i_bm);
    
    // solve poisson equation
    poisson_solver(1.0e-3, d_rho, d_phi);
    
    // derive electric fields from potential
    field_solver(d_phi, d_Ex, d_Ey);
    
    // move particles
    particle_mover(d_e, d_e_bm, d_i, d_i_bm, d_Ex, d_Ey);
    
    // contour condition
    cc(t, d_e_bm, &d_e, d_i_bm, &d_i, d_Ex, d_Ey);

    if (i>=n_prev && i%n_save==0) {
      sprintf(filename, "e_t_%d", i);
      particles_snapshot(d_e, d_e_bm, filename);
      sprintf(filename, "i_t_%d", i);
      particles_snapshot(d_i, d_i_bm, filename);
      sprintf(filename, "charge_t_%d", i);
      mesh_snapshot(d_rho, filename);
      sprintf(filename, "potential_t_%d", i);
      mesh_snapshot(d_phi, filename);
    }
  }
  
  cout << "Simulation finished!" << endl;
  return 0;
}
