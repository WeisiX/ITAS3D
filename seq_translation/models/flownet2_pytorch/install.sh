#!/bin/bash
cd ./networks/correlation_package
chmod u+x make.sh
./make.sh
cd ../resample2d_package 
chmod u+x make.sh
./make.sh
cd ../channelnorm_package 
chmod u+x make.sh
./make.sh
cd ..
