#!/bin/bash

# Number possible groups
num_files=40

# vars
PORT=22000
MACHINE=paffenroth-23.dyn.wpi.edu

# Loop to create files
for i in $(seq 1 $num_files); do
    echo "trying group ${i} at port $((${i}+${PORT})) "
    echo "------------------------------------------------"
    echo "------------------------------------------------"
    ssh -i ../../scripts/student-admin_key -p $((${i}+${PORT})) -o StrictHostKeyChecking=no student-admin@${MACHINE} hostname
    echo "------------------------------------------------"
    echo "------------------------------------------------"    
done


