/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/


#ifndef STD_H
#define STD_H

/****************************** HEADERS ******************************/

#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <fstream>

using namespace std;

/************************ SYMBOLIC CONSTANTS *************************/

#define PI 3.1415926535897932   //symbolic constant for PI

/************************ PARTICLE STRUCTURE *************************/

struct particle
{
  double x;
  double y;
  double vx;
  double vy;
};


#endif
