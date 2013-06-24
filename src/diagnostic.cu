/****************************************************************************
 *                                                                          *
 *    This file is part of CUPIC, a code that simulates the interaction     *
 *    between plasma and a langmuir probe using PIC techniques accelerated  *
 *    with the use of GPU hardware (CUDA extension of C/C++)                *
 *                                                                          *
 ****************************************************************************/

/****************************** HEADERS ******************************/

#include "diagnostic.h"

/********************* HOST FUNCTION DEFINITIONS *********************/

int number_of_particles(int *d_bm) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int ncy = init_ncy();      // number of cells in y dimension
  
  int h_bm[2*ncy];
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // copy vector of bookmarks from device to host
  cudaMemcpy (h_bm, d_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);
  
  return h_bm[2*ncy-1]-h_bm[0]+1;
}


/**********************************************************/

void snapshot(particle *d_p, int * d_bm, string filename) 
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory  
  particle *h_p;
  int N;
  ofstream file;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // evaluate number of particles in the vector
  N = number_of_particles(d_bm);
  
  // allocate host memory for particle vector
  h_p = (particle *) malloc(N*sizeof(particle));
  
  // copy particle vector from device to host
  cudaMemcpy (h_p, d_p, N*sizeof(particle), cudaMemcpyDeviceToHost);
  
  // save snapshot to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  file.open(filename.c_str());
  
  for (int i = 0; i < N; i++) 
  {
    file << i << " " << h_p[i].x << " " << h_p[i].y << " " << h_p[i].vx << " " << h_p[i].vy << endl;
  }
  
  file.close();
  
  return;
}

/**********************************************************/

void show_bm(int * d_bm)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const int ncy = init_ncy();      // number of cells in y dimension
  int h_bm[2*ncy];
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // copy vector of bookmarks from device to host
  cudaMemcpy (h_bm, d_bm, 2*ncy*sizeof(int), cudaMemcpyDeviceToHost);

  // print bookmarks
  cout << "| ";
  for (int i = 0; i<2*ncy; i+=2)
  {
    cout << h_bm[i] << "," << h_bm[i+1] << " | ";
  }
  cout << endl;

  return;
}

/**********************************************************/

void save_bins(int *d_bm, particle *d_p, string filename)
{
  /*--------------------------- function variables -----------------------*/
  
  // host memory
  static const double ds = init_ds();      // spacial step
  particle *h_p;
  int N;
  ofstream file;
  
  // device memory
  
  
  /*----------------------------- function body -------------------------*/
  
  // evaluate number of particles in the vector
  N = number_of_particles(d_bm);
  
  // allocate host memory for particle vector
  h_p = (particle *) malloc(N*sizeof(particle));
  
  // copy particle vector from device to host
  cudaMemcpy (h_p, d_p, N*sizeof(particle), cudaMemcpyDeviceToHost);
  
  // save bins to file
  filename.insert(0, "../output/");
  filename.append(".dat");
  file.open(filename.c_str());
  
  for (int i = 0; i < N; i++) 
  {
    file << i << " " << int(h_p[i].y/ds) << endl;
  }
  
  file.close();
  
  return;
}

/******************** DEVICE KERNELS DEFINITIONS *********************/



/**********************************************************/
