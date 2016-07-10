# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <omp.h>

int main ( int argc, char *argv[] );
/******************************************************************************/

int main ( int argc, char *argv[] )

/******************************************************************************/
/*
*/
{
# define M 12
# define N 12
# define ITER 1000

  int i, j, cur, temp_i, temp_j;
  
  double epsilon = 0.001;
  double mean = 0.0;
  double diff, my_diff;
  

  double u[M][N];

/*
* Begin setup of the array. 
*/
  #pragma omp parallel shared( u ) private(i, j) reduction(+ : mean)
  {
    #pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      u[i][0] = 100.0;
    }
    #pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      u[i][N-1] = 100.0;
    }
    #pragma omp for
    for ( j = 0; j < N; j++ )
    {
      u[M-1][j] = 100.0;
    }
    #pragma omp for
    for ( j = 0; j < N; j++ )
    {
      u[0][j] = 0.0;
    }
/*
  Average the boundary values, to come up with a reasonable
  initial value for the interior.
*/
    #pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + u[i][0];
    }
    #pragma omp for
    for ( i = 1; i < M - 1; i++ )
    {
      mean = mean + u[i][N-1];
    }
    #pragma omp for
    for ( j = 0; j < N; j++ )
    {
      mean = mean + u[M-1][j];
    }
    #pragma omp for
    for ( j = 0; j < N; j++ )
    {
      mean = mean + u[0][j];
    }
  }
  mean = mean / ( double ) ( 2 * M + 2 * N - 4 );
  printf ( "\n" );
  printf ( "  MEAN = %f\n", mean );
/* 
  Initialize the interior solution to the mean value.
*/
  #pragma omp parallel shared( u ) private(i, j)
  {
    #pragma omp for
    for(i = 1; i < M -1; i++)
    {
      for(j = 1; j < N -1; j++)
      {
        u[i][j] = mean;
      }
    }
  }

  printf(" MEAN = %f\n", mean);

  /*
  * End array setup so at this point our array contains the values
  * that it starts with.
  */
  diff = epsilon;  
  int iteration_number = 0;
  int run = 1;
  double wtime = omp_get_wtime();
  while(run)
  {
    int cont = 0;
    my_diff = 0.0;
    printf("Currently running on iteration number %d with diff %f\n", iteration_number, diff);
    diff = 0.0;
    iteration_number++;
    #pragma omp parallel shared(u, diff) private(i, j, cur, mean, temp_i, temp_j) reduction(+ : cont)
    {
      srand((int)time(NULL) ^ omp_get_thread_num());
      for(i = 1; i < M-1; i++)
      {
        #pragma omp for
        for(j = 1; j < N-1; j++)
        {
          mean = 0.0;
          for(cur = 0; cur < ITER; cur++)
          {
            temp_i = i;
            temp_j = j;
            while(1)
            {
              int direction = rand()%4;
              //Go towards the i = 0 row 
              if(direction == 0)
              {
                temp_i--;
                if(temp_i == 0){mean += 0.0; break;}
              }
              //Go towards the j = 0 col 
              else if(direction == 1)
              {
                temp_j--;
                if(temp_j == 0){mean += 100.0; break;}
              }
              //Go towards the i = M row 
              else if(direction == 2)
              {
                temp_i++;
                if(temp_i == (M-1)){mean += 100.0; break;}
              }
              //Go towards the j = N col 
              else
              {
                temp_j++;
                if(temp_j == (N-1)){mean += 100.0; break;}
              }
            }
          }
          double old = u[i][j];
          if(iteration_number == 0)
          {
             u[i][j] = (double) (u[i][j] + mean)/(ITER + 1);
          }
          else
          {
            double cur_iter = (double) iteration_number * ITER;
            double prev_avg = (double) cur_iter * u[i][j];
            u[i][j] = (double) (prev_avg + mean) / (cur_iter + ITER); 
          }
          if( fabs(old - u[i][j])  > epsilon)
          {
            if( fabs(old - u[i][j]) > my_diff)
            {
              my_diff = fabs(old - u[i][j]);
            }
            cont++;
          }
        }
      }
    #pragma omp critical
    {
      if(my_diff > diff){diff = my_diff;}
    }
    }
    if(cont == 0){run = 0;}
  }
  wtime = omp_get_wtime() - wtime;
  printf("Time taken %f\n", wtime);
  return 0;

# undef M
# undef N
}
    /*
    for(i = 0; i < M; i++)
    {
      for(j = 0; j < N; j++)
      {
        printf("%f ", u[i][j]);
      }
      printf("\n");
    }
    */
