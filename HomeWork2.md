
# CSE 566S Homework 2
## Christopher Ogle
___

## Monte Carlo Algorithm Analysis
___

<p>
For this particular assignment we were tasked with creating a Monte Carlo
Version of the provided algorithm. For my implementation the time that the
algorithm takes to converge is painstakingly slow. Much, much slower than the
Gradient Descent algorithm. I sort of expected it to be slow, but I did not
expect it to be this slow. Below is a snippet of the critical section of my
code and following that is an explanation of why my version is so slow.
</p>


```python
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
```

<p>
This snippet of code is the main logic behind my algorithm. The reason that this
particular piece of code takes so much longer is that we are using a
non-deterministic method in order to try and solve the problem. While the
gradient descent iterates a definite amounts of times per iteration. The
Monte Carlo has no such upper limit. Given that each time through the algorithm
it must iterate until it reaches an edge those points in the middle will take
a large amount of time in order to complete their journey to the edge. In
addition as we increase the size of the graph this only makes the algorithm have
to work more. Given that points in the center of graph are further away from the
edge it will take much longer to reach the edge. 
</p>
<p>
Another thing that really is difficult with this algorithm is that the
difference between the previous value at the new value may jump
around quite a bit. Because this does not always decrease it can be a real pain
to have to run this algorithm. In fact I think if I tried it on a large array like
40 by 40, I would most likely overflow the numbers as I am taking the weighted
average and calculation how many previous iteration I have taken would be a
very large number. One thing that can be done to try and combat this to increase
the number of times that you run the algorithm per given iteration. While this
helps due to the Law of Large Numbers again you run into the problem of this
algorithm, as written, taking a long time, increasing the number of iterations 
you take in the while loop will nondeterministically increase number of 
iterations of the algorithm.
</p>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/Monte/Flucation.PNG"></img>
<i>For a simple 12 by 12 grid with an <b>ITER</b> value of 1000</i>

<p>
Looking at this picture we can see just how much slower this implementation of
the algorithm is the small graph takes nearly 13 minutes to complete. In
addition the large variance between runs can be seen, which really inhibits one
from determining when this algorithm will reach termination. If one has the time
and is trying to simulation the randomness of particles then, this method might
be the best; however I would prefer the provided code over this algorithm.
</p>

## Problem 3
___
<center><h3>Data Collection Methodology</h3></center>
<p>
An important thing to discuss is how data was collected for the following
trials. The code that is being used for this problem is the supplied code
given to us in the zip file. For this particular test the 
<b>OMP_NUM_THREADS</b> variable is set to four, further exploration of the
how the number of threads affects run time will be looked at later in
Problem number 5. Each test will be ran five times with our timing results
based off of the timing hook provided in the code (explanation of why in conclusion).
All test were run from the school's system. 
</p>


```python
import plotly.plotly as py
import plotly.graph_objs as go

#CPU events registered, pulled from results folder
y0 = [13.530811, 13.134220, 12.986208, 12.984937, 13.180286]
y1 = [6.335300, 6.365037, 6.335672, 6.375288, 6.640547]
y2 = [3.154183, 3.151281, 3.294024, 3.160028, 3.361889]
y3 = [2.982923, 3.163661, 3.316781, 3.000088, 2.992154]
y4 = [2.485518, 2.546888, 2.478904, 2.492690, 2.630998]

#Each Traces represents a level of optimization
trace0 = go.Box(
    y=y0,
    name = 'No Opt'
)
trace1 = go.Box(
    y=y1,
    name = 'Level 1 Opt'
)
trace2 = go.Box(
    y=y2,
    name = 'Level 2 Opt'
)
trace3 = go.Box(
    y=y3,
    name = 'Level 3 Opt'
)
trace4 = go.Box(
    y=y4,
    name = 'Fast Opt'
)
data = [trace0, trace1, trace2, trace3,trace4]

layout=go.Layout(height=750,
                 title="OpenMP: Time to run vs Optimization", 
                 xaxis={'title':'Optimization Level'}, 
                 yaxis={'title':'Time (sec)'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='OpenMP_graph')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/20.embed" height="750px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/20/" target="_blank" 
       title="OpenMP: Time to run vs Optimization" 
       style="display: block; text-align: center;">
       
        <img src="https://plot.ly/~cogle/20.png" alt="OpenMP: Time to run vs Optimization" 
             style="max-height:1000"  
             onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
    </a>
    <script data-plotly="cogle:20"  src="https://plot.ly/embed.js" async></script>
</div>

<center><h3>OpenMP Optimization Data Analysis</h3></center>

<p>
Looking at the graph above we can clearly see the trend that as you increase the
optimization level the time required to run the algorithm decreases as well too.
In the previous assignment we saw that as the optimization level increased, that
didn't always correspond in an increase in time.
As stated above this particular experiment used 4 threads; one of the things
that using 4 threads allows us to do is that we can take advantage assigning
each thread to its own core. This allows each thread to take advantage of cache
locality when iterating through the for loops. In addition when the compiler
optimizes the code each OpenMP thread receives the optimized code. This allows
each of the threads to run that optimized code, and results in further speed 
ups. 
</p>

<p>
In order to try and determine what lead to such a drop in the speed I compiled
the program at optimization level one and then using the following site
<a>https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html</a> I applied
each one the optimizations to the program and made it. While lots of
optimizations decreased the time bit by bit I noticed that the
<b>-fcaller-saves</b> optimization decreased the amount of time that the code
to run by about 2 seconds from what we see as the average time to run code
at Level 1 Optimization.
<p>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/OptimizationSnip/Timing.PNG"></img>
<p><i>With <b>-fcaller-saves</b></i><p>

<h2>Problem 4</h2>
___
<center><h3>Data Collection Methodology</h3></center>
<p>
For this particular problem the data was collected by running the supplied code
compiled with optimization level two on the schools system with 4 threads. As 
before the basis of our  time measurement comes from the in code timing hooks. 
</p>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

# Create random data with numpy
import numpy as np

x = ['50', '100', '150' ,'200', '250',
     '300', '400', '500' ,'600', '700',
     '800', '900', '1000' ,'1100', '1200',
     '1300', '1400', '1500']


size_50 = [0.014645,0.014547,0.014831,0.014647,0.014675]
size_100 = [0.065983,0.066967,0.066720,0.068457,0.064742]
size_150 = [0.147041,0.148926,0.155783,0.151992,0.154726]
size_200 = [0.336146,0.333026,0.330973,0.333543,0.335950]
size_250 = [0.615197,0.621354,0.614993,0.615967,0.612557]
size_300 = [0.987381,0.984063,0.965989,0.964826,0.969361]
size_400 = [1.948594,1.932207,1.939529,1.938243,1.944792]
size_500 = [3.216002,3.201695,3.212385,3.561511,3.218516]
size_600 = [4.700759,4.745719,4.745413,4.839352,4.810836]
size_700 = [6.468956,6.435122,6.446439,6.355941,6.335129]
size_800 = [9.214062,9.086559,8.918791,9.023419,8.997899]
size_900 = [10.900986,11.079394,10.756924,10.719667,10.764894]
size_1000 = [13.107303,13.938686,13.401571,13.119014,13.222637]
size_1100 = [15.861555,15.932275,16.226205,15.939537,16.243901]
size_1200 = [19.551888,19.000210,19.613453,19.400393,18.937407]
size_1300 = [23.125764,24.162420,23.157251,36.779754,23.727463]
size_1400 = [47.653508,39.049947,39.391193,41.874674,47.111895]
size_1500 = [57.264617,55.271507,58.151817,57.803165,58.615409]



y0 = [np.average(size_50),
      np.average(size_100),
      np.average(size_150),
      np.average(size_200),
      np.average(size_250),
      np.average(size_300),
      np.average(size_400),
      np.average(size_500),
      np.average(size_600),
      np.average(size_700),
      np.average(size_800),
      np.average(size_900),
      np.average(size_1000),
      np.average(size_1100),
      np.average(size_1200),
      np.average(size_1300),
      np.average(size_1400),
      np.average(size_1500)
     ]


# Create traces
trace0 = go.Scatter(
    x = x,
    y = y0,
    mode = 'lines+markers',
    name = 'r = vM (Default)',
    error_y=dict(
        type='data',
        array=[np.std(size_50), 
               np.std(size_100),
               np.std(size_150),
               np.std(size_200),
               np.std(size_250),
               np.std(size_300),
               np.std(size_400),
               np.std(size_500),
               np.std(size_600),
               np.std(size_700),
               np.std(size_800),
               np.std(size_900),
               np.std(size_1000),
               np.std(size_1100),
               np.std(size_1200),
               np.std(size_1300),
               np.std(size_1400),
               np.std(size_1500)],
        visible=True
    )
)

data = [trace0]
layout=go.Layout(height=1000,
                 title="OpenMP: Time to run vs Array Size", 
                 xaxis={'title':'Size of Array (N x N)'}, 
                 yaxis={'title':'Time(sec)'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Thread-Timing-Array')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/24.embed" height="1000px" width="100%"></iframe>



<center><h3>Hypothesis</h3></center>
<p>
The provided algorithm’s time complexity is O(n^2); with this knowledge we can predict that our graph should resemble a parabolic curve.
</p>

<div>
    <a href="https://plot.ly/~cogle/24/" target="_blank" 
       title="OpenMP: Time to run vs Number of Threads" 
       style="display: block; text-align: center;">
       
        <img src="https://plot.ly/~cogle/24.png" alt="OpenMP: Time to run vs Number of Threads" 
             style="max-height:1000"  
             onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
    </a>
    <script data-plotly="cogle:22"  src="https://plot.ly/embed.js" async></script>
</div>

<center><h3>OpenMP Array Size Dependence Data Analysis</h3></center>
<p>
From the graph we see exactly what we predicted in our Hypothesis. It is really nice to see that our algorithm follows hypothesis. With the introduction of threads some might think that our algorithm should run faster; however, the way our program works is we divide up the work amongst the threads thus as we increase the size of the array each thread receives n^2 more work.  The results that we saw weren’t too surprising and fell in line with what was expected.
</p>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

# Create random data with numpy
import numpy as np

x = ['1.0', '1.0^-1', '1.0^-2' ,'1.0^-3', '1.0^-4',
     '1.0^-5', '1.0^-6', '1.0^-7' ,'1.0^-8']


ep_1 = [0.008293,0.008271,0.008236,0.008312,0.008446]
ep_01 = [0.066485,0.065135,0.063764,0.063592,0.064369]
ep_001 = [0.416287,0.414127,0.417072,0.413306,0.412400]
ep_0001 = [3.464793,3.223525,3.149614,3.224029,3.139216]
ep_00001 = [10.518056,10.734608,10.768388,10.229248,10.420260]
ep_000001 = [18.615514,19.272254,18.676508,18.874452,18.847971]
ep_0000001 = [26.936042,26.929934,27.576725,27.761543,27.062500]
ep_00000001 = [38.918818,36.855797,38.565591,36.609836,36.874943]
ep_000000001 = [49.057643,50.395065,47.806317,49.886437,49.790664]


y0 = [np.average(ep_1),
      np.average(ep_01),
      np.average(ep_001),
      np.average(ep_0001),
      np.average(ep_00001),
      np.average(ep_000001),
      np.average(ep_0000001),
      np.average(ep_00000001),
      np.average(ep_000000001)
     ]


# Create traces
trace0 = go.Scatter(
    x = x,
    y = y0,
    mode = 'lines+markers',
    name = 'r = vM (Default)',
    error_y=dict(
        type='data',
        array=[np.std(ep_1), 
               np.std(ep_01),
               np.std(ep_001),
               np.std(ep_00001),
               np.std(ep_000001),
               np.std(ep_0000001),
               np.std(ep_00000001),
               np.std(ep_000000001)
              ],
        visible=True
    )
)

data = [trace0]
layout=go.Layout(height=1000,
                 title="OpenMP: Time to run(500x500 4 Threads) vs Epsilon Precision", 
                 xaxis={'title':'Value of Epsilon'}, 
                 yaxis={'title':'Time(sec)'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Epsilon-Timing-Array')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/26.embed" height="1000px" width="100%"></iframe>



<center><h3>Hypothesis</h3></center>
<p>
I am not 100% sure how this graph will look but since for this problem I am reducing epsilon by a power of ten each time so it might resemble a logarithmic graph.
</p>

<div>
    <a href="https://plot.ly/~cogle/26/" target="_blank" 
       title="OpenMP: Time to run(500x500 4 Threads) vs Epsilon Precision" 
       style="display: block; text-align: center;">
       
        <img src="https://plot.ly/~cogle/26.png" alt="OpenMP: Time to run(500x500 4 Threads) vs Epsilon Precision" 
             style="max-height:1000"  
             onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
    </a>
    <script data-plotly="cogle:22"  src="https://plot.ly/embed.js" async></script>
</div>

<center><h3>OpenMP Epsilon Value Dependence Data Analysis</h3></center>
<p>
From the graph we see that while it does resemble something of a logarithmic graph without an equation we cannot be 100%. Of course though the results are as expected as we decrease the value of epsilon the number of iterations that the algorithm will require to converge will be greater as we are demanding more precision from the algorithm thus our time for termination will take longer.
</p>

<h2>Problem 5</h2>
___
<center><h3>Data Collection Methodology</h3></center>
<p>
For this particular problem the data was collected by running the supplied code
compiled with optimization level two on the schools system. As before the basis
of our  time measurement comes from the in code timing hooks. 
</p>


```python
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF 

# Create random data with numpy
import numpy as np

x = ['1', '2', '3' ,'4', '5',
     '6', '7', '8' ,'9', '10',
     '11', '12', '13' ,'14', '15',
     '16', '17', '18' ,'20', '24',
     '28', '32']


thread_1 = [10.898600,12.290995,10.954756,11.661482,10.960365]
thread_2 = [5.894024,5.890253,5.977245,5.880431,5.882158]
thread_3 = [4.176541,4.108237,4.116009,4.180820,4.174366]
thread_4 = [3.217831,3.153855,3.329593,3.408588,3.166579]
thread_5 = [2.730287,2.711400,2.710112,2.709504,2.642026]
thread_6 = [2.334213,2.334365,2.325414,2.451602,2.383726]
thread_7 = [2.117244,2.109290,2.112163,2.134355,2.140245]
thread_8 = [1.965674,1.970861,1.950640,1.981355,1.984397]
thread_9 = [1.897913,1.825782,1.842638,1.822134,1.840135]
thread_10 = [1.806611,1.774046,1.744866,1.797151,1.770225]
thread_11 = [1.734292,1.682412,1.670571,1.630883,1.689738]
thread_12 = [1.657432,1.627978,1.628685,1.635920,1.613842]
thread_13 = [144.503488,90.712890,107.803704,145.356468,86.514884]
thread_14 = [202.512106,204.469967,203.908372,203.810552,204.043968]
thread_15 = [208.167857,207.126849,208.316489,207.540062,209.958652]
thread_16 = [213.714671,213.416896,213.798919,213.025954,213.123323]
thread_17 = [4.224755,4.252676,3.967282,3.952878,4.039162]
thread_18 = [3.990004,3.962726,4.197004,4.090027,3.979533]
thread_20 = [4.148234,4.108585,4.097461,4.474365,4.185891]
thread_24 = [4.616228,4.576650,4.988805,4.596670,4.568135]
thread_28 = [5.245399,5.319254,5.572317,4.870642,5.495162]
thread_32 = [5.438613,6.375927,5.940444,5.687882,5.948909]
thread_36 = [6.530933,6.799963,6.203396,6.301346,7.821452]


y0 = [np.average(thread_1),
      np.average(thread_2),
      np.average(thread_3),
      np.average(thread_4),
      np.average(thread_5),
      np.average(thread_6),
      np.average(thread_7),
      np.average(thread_8),
      np.average(thread_9),
      np.average(thread_10),
      np.average(thread_11),
      np.average(thread_12),
      np.average(thread_13),
      np.average(thread_14),
      np.average(thread_15),
      np.average(thread_16),
      np.average(thread_17),
      np.average(thread_18),
      np.average(thread_20),
      np.average(thread_24),
      np.average(thread_28),
      np.average(thread_32),
      np.average(thread_36),
     ]


# Create traces
trace0 = go.Scatter(
    x = x,
    y = y0,
    mode = 'lines+markers',
    name = 'r = vM (Default)',
    error_y=dict(
        type='data',
        array=[np.std(thread_1), 
               np.std(thread_2),
               np.std(thread_3),
               np.std(thread_4),
               np.std(thread_5),
               np.std(thread_6),
               np.std(thread_7),
               np.std(thread_8),
               np.std(thread_8),
               np.std(thread_9),
               np.std(thread_10),
               np.std(thread_11),
               np.std(thread_12),
               np.std(thread_13),
               np.std(thread_14),
               np.std(thread_15),
               np.std(thread_16),
               np.std(thread_17),
               np.std(thread_18),
               np.std(thread_20),
               np.std(thread_24),
               np.std(thread_28),
               np.std(thread_32),
               np.std(thread_36)],
        visible=True
    )
)

data = [trace0]
layout=go.Layout(height=1000,
                 title="OpenMP: Time to run vs Number of Threads", 
                 xaxis={'title':'Number of Threads'}, 
                 yaxis={'title':'Time(sec)'})

figure=go.Figure(data=data,layout=layout)
py.iplot(figure, filename='Thread-Timing')
```




<iframe id="igraph" scrolling="no" style="border:none;" seamless="seamless" src="https://plot.ly/~cogle/22.embed" height="1000px" width="100%"></iframe>



<div>
    <a href="https://plot.ly/~cogle/22/" target="_blank" 
       title="OpenMP: Time to run vs Number of Threads" 
       style="display: block; text-align: center;">
       
        <img src="https://plot.ly/~cogle/22.png" alt="OpenMP: Time to run vs Number of Threads" 
             style="max-height:1000"  
             onerror="this.onerror=null;this.src='https://plot.ly/404.png';" />
    </a>
    <script data-plotly="cogle:22"  src="https://plot.ly/embed.js" async></script>
</div>

<p>
It would appear that as we up until 12 threads we were getting the results that
we expected as more threads were added a gradual decrease in time take was
achieved. However, when using 13 to 16 threads the number of time it takes
skyrockets. Using the <b>lscpu</b> command we can determine that our CPU is a
Intel® Xeon® Processor E5-2630 v3 (20M Cache, 2.40 GHz), from the manufacturer's
spec sheet we that this has 8 physical cores while supporting 16 threads;
through hyperthreading each core has two threads. Using the Scalasca utility I
was able to determine which lines the thread was spending most of its time. The
screen shot below is from the results I gathered. 
</p>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/NoWaitSnip/NoWait.PNG"></src>
<i>Omp for with nowait 14 Threads</i>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/NoWaitSnip/Wait.PNG"></img>
<i>Omp for without nowait 14 Threads(Code as originally provided)</i>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/NoWaitSnip/10_threads.PNG"><img>
<i>Omp for running 10 Threads</i>

<p>
From the above screenshots we see that indeed making the code not wait did have
a significant difference. However, it did not speed it up to the speeds of the
ten thread run. So while we were able to speed up the speed a good amount we
still see that there is a lot of difference between running the code with 14
threads and 10 threads. This leads me to believe that the code is not being
optimized well by OpenMP. Another interesting observation is that the poor
timing continues up until the maximum number of threads has been reached, and
then immediately after that the numbers return a decent level. I would suspect
that there is sub-optimal distribution of the work going on with the threads.
</p>
<p>
In addition one of the things that I thought might be hindering the performance
is that as we increase the number of threads and each core now shares multiple
threads the cache will get written over much more frequently resulting in more
cache misses. This doesn't explain the results that we saw, with there being 
a sharp jump in time, but it could be a contributing factor. 
</p>

<center><h2>Conclusion</h2></center>
<p>
Overall the greatest difficulty I had with this problem was the Monte Carlo Algorithm. I am still not 100% my implementation is correct or optimal. The algorithm runs very slow and is very, computational intensive. In addition, the random movements that this algorithm required introduces a real source of uncertainty into the time complexity. It is my belief that because of the random walking our algorithm falls short compared to the Gradient Descent algorithm. In that algorithm it simply has to loop over and average the four corners. Our algorithm has to make its way to a corner and then repeat this process multiple times. This is a real time suck. Finding our way from inner part of the matrix to an outer edge, from the center, will at best take 250 steps. Then we must repeat this process an undefined number of times. Thus the time complexity of the algorithm quickly blows up at this step. Another thing I found that is the value of the of epsilon in the in the Monte Carlo algorithm is really volatile. Because of the way the algorithm works I feel that values in the center have the most volatility. While those values at the edges will most likely reach a stable value rather quickly the values in the center may in one given run go to the top run and then on the next run may go to left or right. This then gives us two different values. While the numbers I believe do converge do to the Law of Large Numbers the time that it takes is very long.   
</p>

<p>
Another thing that I noticed was that when using Scalasca in order to try and 
determine how long a program took I was outputting various time for how long
the program actually took to run. 
For instance using the command line and asking how long the program ran resulted 
in this measurement being given.
</p>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/Scalasca/TimeDifference.PNG"></img>

<p>However opening up the GUI gave this measurement</p>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/Scalasca/TimeDifferenceOnScore.PNG"></src>

<p>Finally the program's critical section only took this long to run</p>

<img src="https://raw.githubusercontent.com/cogle/CSE566_Homework_2/master/Results/Scalasca/ProgramOutput.PNG"></src>

<p>
Each of the Scalasca's time is a different value, and much longer than how long
the actual program ran. Because of these discrepancies I relied on the timing
hooks within the code as the basis of my measurements.
</p>


```python

```
