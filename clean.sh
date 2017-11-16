#!/bin/bash

path="."
if [ "$#" -ne 0 ]; then
    path=$1
fi

rm -v $path/*.*zn
