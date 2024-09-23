#! /bin/bash

PORT=21002
MACHINE=paffenroth-23.dyn.wpi.edu
LOCAL_USER=simeon_krastev
DEFAULT_KEY_NAME=ta_key
MACHINE_USER=ta

# Clean up from previous runs
ssh-keygen -f "/home/${LOCAL_USER}/.ssh/known_hosts" -R "[${MACHINE}]:${PORT}"
rm -rf tmp

# Create a temporary directory
mkdir tmp

# copy the key to the temporary directory
cp ${DEFAULT_KEY_NAME}* tmp

# Change to the temporary directory
cd tmp

# Set the permissions of the key
chmod 600 ${DEFAULT_KEY_NAME}*

# Create a unique key
rm -f my_key*
ssh-keygen -f my_key -t ed25519 -N "careful"

# Insert the key into the authorized_keys file on the server
# One > creates
cat my_key.pub > authorized_keys
# two >> appends
cat ${DEFAULT_KEY_NAME}.pub >> authorized_keys
chmod 600 authorized_keys

echo "checking that the authorized_keys file is correct"
ls -l authorized_keys
cat authorized_keys

# Copy the authorized_keys file to the server
scp -i ${DEFAULT_KEY_NAME} -P ${PORT} -o StrictHostKeyChecking=no authorized_keys ${MACHINE_USER}@${MACHINE}:~/.ssh/

# Add the key to the ssh-agent
eval "$(ssh-agent -s)"
ssh-add my_key

# Check the key file on the server
echo "checking that the authorized_keys file is correct"
ssh -i ${DEFAULT_KEY_NAME} -p ${PORT} -o StrictHostKeyChecking=no ${MACHINE_USER}@${MACHINE} "cat ~/.ssh/authorized_keys"

# Clean old versions if present
ssh -i ${DEFAULT_KEY_NAME} -p ${PORT} -o StrictHostKeyChecking=no ${MACHINE_USER}@${MACHINE} "rm log.txt && rm -rf CS553_example"

# clone the repo
git clone https://github.com/rcpaffenroth/CS553_example
git checkout auto_deployment_test

# Copy the files to the server
scp -P ${PORT} -o StrictHostKeyChecking=no -r CS553_example ${MACHINE_USER}@${MACHINE}:~/

# check that the code in installed
COMMAND="ssh -i my_key -p ${PORT} -o StrictHostKeyChecking=no ${MACHINE_USER}@${MACHINE}"

${COMMAND} "ls CS553_example"
${COMMAND} "sudo apt install -qq -y python3-venv"
${COMMAND} "cd CS553_example && python3 -m venv venv"
${COMMAND} "cd CS553_example && source venv/bin/activate && pip install -r requirements.txt"
${COMMAND} "nohup CS553_example/venv/bin/python3 CS553_example/app.py > log.txt 2>&1 &"

# nohup ./whatever > /dev/null 2>&1 

# debugging ideas
# sudo apt-get install gh
# gh auth login
# requests.exceptions.HTTPError: 429 Client Error: Too Many Requests for url: https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta/v1/chat/completions
# log.txt
