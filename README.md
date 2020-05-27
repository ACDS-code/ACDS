# ACDS
This repository contains all necessary code from 'Gaussian Process Assisted Active Learning of Physical Laws'.

Table 2 can be recovered by code from ACDS-code/ACDS/ode.
In this folder, there are two main functions:
              ode.m is ACDS method;
              space_filling.m is maximin space-filling method.

Table 3 can be recovered by code from ACDS-code/ACDS/ode. All codes are same with Table 2.

Table 4 can be recovered by code from ACDS-code/ACDS/ode_rand_cof.
In this folder, there are two main functions:
              ode.m is ACDS method;
              space_filling.m is maximin space-filling method.
             
Table 5 can be recovered by code from ACDS-code/ACDS/Bass.
In this folder, there are two main functions:
              ode.m is ACDS method;
              space_filling.m is maximin space-filling method.       
              
Table 10 can be recovered by code from ACDS-code/ACDS/air_pollution.
In this folder, there are three main functions:
              advection_dispersion.m is ACDS method;
              space_filling.m is maximin space-filling method;
              D_optimal.m is D_optimality method.
              
              
##### Some remarks (common for all models)
code relating to GP: gp_new.m 
                     se_kernel.m
                     se_kernel_gradient.m
                     se_kernal_hessian.m
                     lml_exact.m
                     minimize_quiet.m

code relating to ACDS: acds.m
