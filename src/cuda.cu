/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "cuda.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

void cu_check(cudaError_t cuError) 
{
  // function variables
  
  
  // function body
  
  if (0 == cuError)
  {
    return;
  } else
  {
    cout << "CUDA error found. (error code: " << cuError << ")" << endl;
    cout << "Exiting simulation" << endl;
    exit(1);
  }
  
}

/**********************************************************/



/******************** DEVICE KERNELS DEFINITIONS *********************/



/**********************************************************/