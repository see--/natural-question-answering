#!/usr/bin/env python3

# adapted from https://raw.githubusercontent.com/GoogleCloudPlatform/training-data-analyst/master/courses/fast-and-lean-data-science/create-tpu-deep-learning-vm.sh  # noqa
# by @martin-gorner

import os
import random

zone = 'europe-west4-a'
instance_name = 'nq2'
image_family = 'tf2-2-1-cpu'
machine_type = 'n1-standard-8'

startup_script = f"""#! /bin/bash
echo \"export TPU_NAME={instance_name}\" > /etc/profile.d/tpu-env.sh
echo \"alias python='python3'\" >> ~/.bashrc
# install python3.6.9
sudo apt-get install -y zlib1g-dev libssl-dev libbz2-dev libsqlite3-dev liblzma-dev
wget https://www.python.org/ftp/python/3.6.9/Python-3.6.9.tgz
tar -xvf Python-3.6.9.tgz
rm Python-3.6.9.tgz
cd Python-3.6.9
./configure
sudo make install
cd ..
sudo pip3 install --upgrade pip
sudo pip3 install --upgrade tensorflow==2.1 tqdm ipython matplotlib pandas sklearn oauth2client google-api-python-client tensorflow-hub  # noqa
sudo pip3 install --upgrade git+https://github.com/huggingface/transformers.git"""

startup_file = 'startup.sh'
with open(startup_file, 'w') as f:
  f.write(startup_script)

vm_command = f"""gcloud compute instances create {instance_name} \\
    --zone={zone} \\
    --machine-type={machine_type} \\
    --image-project=deeplearning-platform-release \\
    --image-family={image_family} \\
    --scopes=cloud-platform \\
    --boot-disk-size=100GB \\
    --boot-disk-type=pd-ssd \\
    --metadata-from-file startup-script={startup_file} \\
    --async"""

print(vm_command)
os.system(vm_command)
os.remove(startup_file)
version = '2.1'
accelerator_type = 'v3-8'
tpu_command = f"""gcloud compute tpus create {instance_name} \\
        --zone={zone} \\
        --network=default \\
        --range=192.168.{random.randint(0, 100)}.0/29 \\
        --version={version} \\
        --accelerator-type={accelerator_type}
"""

print(tpu_command)
os.system(tpu_command)
