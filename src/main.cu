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


/****************************** MAIN FUNCTION ******************************/

int main (int argc, const char* argv[])
{
  
  // host variables definition
  double **h_qi, *h_qe, *h_mi, *h_me, *h_kti, *h_kte;  //properties of particles
  

  // device variables definition
  double *d_qi, *d_qe, *d_mi, *d_me, *d_kti, *d_kte;  //properties of particles


  gsl_rng_env_setup();
  static gsl_rng * rng;
  rng = gsl_rng_alloc(gsl_rng_default);  //default random number generator (gsl)

  cout << gsl_rng_uniform_pos(rng) << endl;

  return 0;
}

/****************************** FUNCTION DEFINITION ******************************/

void read_particle_properties (double *h_qi, double *h_qe, double *h_mi, double *h_me, double *h_kti, double *h_kte)
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
  } else
  {
    cout << "input data file could not be opened" << endl;
  }

  return; 
}
