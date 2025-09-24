#! /bin/bash

PORT=21003
MACHINE=paffenroth-23.dyn.wpi.edu

ssh -p ${PORT} -o StrictHostKeyChecking=no rcpaffenroth@${MACHINE}
