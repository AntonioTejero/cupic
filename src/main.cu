/****************************************************************************
 *                                                                          *
 *    CUPIC is a code that simulates the interaction between plasma and     *
 *    a langmuir probe using PIC techniques accelerated with the use of     *
 *    GPU hardware (CUDA extension of C/C++)                                *
 *                                                                          *
 ****************************************************************************/


#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <gsl/gsl_rng.h>			//gsl library for random number generation
#include <gsl/gsl_randist.h>		//gsl library for random number generation

using namespace std;

#define PI 3.1415926535897932		//symbolic constant for PI

int main (int argc, const char* argv[])
{
  gsl_rng_env_setup();
  static gsl_rng * rng;
  rng = gsl_rng_alloc(gsl_rng_default);  //default random number generator (gsl)

  cout << gsl_rng_uniform_pos(rng) << endl;

  return 0;
}
