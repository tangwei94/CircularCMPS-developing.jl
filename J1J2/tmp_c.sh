#!/bin/bash 

if [ ${1} -eq 1 ] ; then
    for t in 0.01 0.02 0.03 0.04 0.05 ; do
        echo ======= branch ${1}, temperature ${t} =======
        julia --project=. J1J2/J1J2_critical.jl ${t}
    done 
elif [ ${1} -eq 2 ] ; then
    for t in 0.06 0.07 0.08 0.09 0.1 ; do
        echo ======= branch ${1}, temperature ${t} =======
        julia --project=. J1J2/J1J2_critical.jl ${t}
    done 
else
    for t in 0.12 0.14 0.16 0.18 0.2  ; do
        echo ======= branch ${1}, temperature ${t} =======
        julia --project=. J1J2/J1J2_critical.jl ${t}
    done 
fi