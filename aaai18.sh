#!/bin/bash

num_cpus=`lscpu | grep '^CPU(s)' | tr -s ' ' | cut -d' ' -f2`

# NOTE make sure to set FACTOR=10 and disable gecode parallelization
for k in `seq 2 5`; do
    for sampling in uniform normal; do
        for c in 0.1 0.5 0.9; do
            for learner in perceptron; do
                echo "RUNNING $k $sampling $c $learner"
                ./aaai18.py exp cartesian \
                    -T 50 \
                    -k $k \
                    -c $c \
                    -X 0.1 0.2 0.5 1 2 5 10 \
                    -L $learner \
                    -W users/utilityParams_synthetic_4_${sampling}.txt -S $sampling \
                    -p $num_cpus \
                    -v \
                    | egrep -i 'user|regret'
            done
        done
    done
done

# NOTE make sure that FACTOR=10 and enable gecode parallelization
for c in 0.1 0.5 0.9; do
    for sampling in uniform normal; do
        for learner in perceptron; do
            echo "RUNNING 2 $sampling $c $learner"
            python3.5 ./aaai18.py exp pc \
                -T 100 \
                -k 2 \
                -c $c \
                -X 0.1 0.2 0.5 1 2 5 10 \
                -L $learner \
                -W users/pc_${sampling}.txt -S ${sampling} \
                -p 1 \
                -M \
                -v \
                | egrep -i 'user|regret'
        done
    done
done

# NOTE make sure that FACTOR=10 and enable gecode parallelization
for k in `seq 2 4`; do
    for c in 0.1 0.5 0.9; do
        for sampling in uniform normal; do
            for learner in perceptron; do
                echo "RUNNING 2 $sampling $c $learner"
                python3.5 ./aaai18.py exp travel \
                    -T 100 \
                    -k $k \
                    -c $c \
                    -X 0.1 0.2 0.5 1 2 5 10 \
                    -L $learner \
                    -S ${sampling} \
                    -p 1 \
                    -M \
                    -v \
                    | egrep -i 'user|regret'
            done
        done
    done
done
