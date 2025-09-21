#! /bin/bash

PORT=21003
MACHINE=paffenroth-23.dyn.wpi.edu
KEY=$HOME/projects/1_classes/DS553_private/scripts/CS2/student-admin_key
ssh -i $KEY -p ${PORT} -o StrictHostKeyChecking=no student-admin@${MACHINE}
