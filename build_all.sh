#!/bin/sh

for i in *.cpp; do
    echo "compiling $i"
        base=`basename $i .cpp`
         g++ -ggdb `pkg-config --cflags opencv` -o `basename $i .cpp` $i `pkg-config --libs opencv`; done

