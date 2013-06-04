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
  double qi = init_qi();                // |
  double qe = init_qe();                // |
  double mi = init_mi();                // |--> particle properties
  double me = init_me();                // |
  double kti = init_kti();              // |
  double kte = init_kte();              // |
  
  double Lx = init_Lx();                // |
  double Ly = init_Ly();                // |--> geometrical properties of simulation
  double ds = init_ds();                // |
  
  double t = 0;                         // time of simulation
  double dt = init_dt();                // time step
  
  double n = init_n();                  // plasma density
  
  double phi_p = init_phi_p();          // probe's potential
  
  double epsilon0 = init_epsilon0();    // electric permitivity

  // device variables definition
  double *d_rho, *d_phi, *d_Ex, *d_Ey;  // properties of mesh (charge density, potential and fields at nodes of the mesh)
  particle *d_e, *d_i;                  // vector of electron and ions
  unsigned int *d_e_bm, *d_i_bm;        // vector that stores the bookmarks (beginning and end point) for each bin (electrons and ions)

  initialize (&d_rho, &d_phi, &d_Ex, &d_Ey, &d_e, &d_i, &d_e_bm, &d_i_bm);
  cout << "1" << endl;
  cout << "number of electrons: " << number_of_particles(d_e_bm) << endl;
  cout << "number of ions: " << number_of_particles(d_e_bm) << endl;
  
  for (int i = 0; i < 10; i++, t += dt) 
  {
    cout << "2" << endl;
    // deposit charge into the mesh nodes
    charge_deposition(d_rho, d_e, d_e_bm, d_i, d_i_bm);
    
    cout << "3" << endl;
    // solve poisson equation
    poisson_solver(1.0e-8, d_rho, d_phi);
    
    cout << "4" << endl;
    // derive electric fields from potential
    field_solver(d_phi, d_Ex, d_Ey);
    
    cout << "5" << endl;
    // move particles
    particle_mover(d_e, d_e_bm, d_i, d_i_bm, d_Ex, d_Ey);
    
    cout << "6" << endl;
    // contour condition
    cc(t, d_e_bm, &d_e, d_i_bm, &d_i, d_Ex, d_Ey);
    cout << "--------------------" << i << endl;
  }
  
  cout << "7" << endl;
  return 0;
}
