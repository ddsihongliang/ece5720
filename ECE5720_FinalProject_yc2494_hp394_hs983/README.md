# ECE5720_Project
ECE5720 Final Project
Author: Hongliang Si, Haodong Ping, Yixin Cui.
Date: 5/15/20190

To Compile:
/usr/local/cuda-10.0/bin/nvcc -arch=compute_52 -o nbody.out main.cu kernel_v1.cu kernel_v2.cu render.cu body.cu

To Run:
./nbody.out <kernel_version>
-where <kernel_version> = 1 or 2

--This will result in 100 .csv files by default. Those files contain position information of N-Bodys.

To See the animation:
Use the MATLAB file attached. 
