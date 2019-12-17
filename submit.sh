#!/bin/sh
for L in 64 128 256 512 1024
do
    for T in 0.8 0.9 0.95 1.0 1.05 1.1 1.2
    do
        condor_submit -b Wolff -append "arguments=${L} ${T}" submit.jds
    done
done
condor_q | grep seyang
