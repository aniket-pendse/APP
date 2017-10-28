#include <stdio.h>
#include <math.h>
#include <float.h>
#include <iostream>
////////////////////////////////////////////////////////////////////////////////
// export C interface
extern "C" 
void computeGold( float* reference, float* idata, const unsigned int len);

////////////////////////////////////////////////////////////////////////////////
//! Compute reference data set
//! Each element is the sum of the elements before it in the array.
//! @param reference  reference data, computed but preallocated
//! @param idata      input data as provided to device
//! @param len        number of elements in reference / idata
////////////////////////////////////////////////////////////////////////////////
void
computeGold( float* reference, float* idata, const unsigned int len) 
{
  reference[0] = 0;
  double total_sum = 0;
  unsigned int i;
  std::cout<<"First data element "<<idata[0]<<std::endl;
  for( i = 0; i < 32; ++i) 
  {
      total_sum += idata[i];
  }
  std::cout<<"Sum of first block "<<total_sum<<std::endl;

  total_sum = 0;
  for( i = len - 33; i < len; ++i) 
  {
      total_sum += idata[i];
  }
  std::cout<<"Sum of last block "<<total_sum<<std::endl;

  total_sum = 0;
  for( i = 0; i < len; ++i) 
  {
      total_sum += idata[i];
  }

  *reference = total_sum;
}

