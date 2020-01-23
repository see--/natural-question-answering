# Overview

In this repository you will find the code for the [TensorFlow 2.0 Question Answering Challenge](https://www.kaggle.com/c/tensorflow2-question-answering). My team finished as 2nd / 1239 with a [micro F1-score](https://www.kaggle.com/c/tensorflow2-question-answering/overview/evaluation) of `0.71` on the private test set. The challenge was to develop better algorithms for natural question answering. You can can find more details about the task and the dataset in these resources:
* https://www.kaggle.com/c/tensorflow2-question-answering/overview
* https://www.kaggle.com/c/tensorflow2-question-answering/discussion/127333
* https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1f7b46b5378d757553d3e92ead36bda2e4254244.pdf
* https://arxiv.org/abs/1901.08634

**Important**: This solution uses [cloud instances](https://cloud.google.com). Don't forget to stop / delete them if you don't need these instances anymore. Unwanted costs might occur otherwise.

- GUI: To stop / delete your VMs go to: [https://console.cloud.google.com](https://console.cloud.google.com) -> `Compute Engine` -> `VM instances`. To stop / delete your TPU instances go to [https://console.cloud.google.com](https://console.cloud.google.com) -> `Compute Engine` -> `TPUs`.

- CLI: To stop your VMs: `sudo shutdown -h now`. To stop your TPU instances: `gcloud compute tpus stop nq2 --zone europe-west4-a`

You can always check your running instances via: `gcloud compute instances list` or using the [console GUI](https://console.cloud.google.com).

# Setup

From a machine that has `gcloud` installed:

```bash
# Download the script to start the instances
wget https://raw.githubusercontent.com/see--/natural-question-answering/master/start_instance.py
python3 start_instance.py
```

This will create a VM named "nq2" with `Python-3.6.9` and all the required libraries installed. It will also create a "v3-8" TPU in the zone "europe-west4-a" with the same name.

For the following commands, please ssh into your VM. You can get the command from [https://console.cloud.google.com](https://console.cloud.google.com) -> `Compute Engine` -> `VM instances` -> `Connect` -> `View gcloud command`.

It should be similar to:
```
gcloud beta compute --project "MYPROJECT" ssh --zone "europe-west4-a" "nq2"
```

# Get the data

```bash
sudo pip install --upgrade kaggle
mkdir .kaggle
# Replace "MYUSER" and "MYKEY" with your credentials. You can create them on:
# `https://www.kaggle.com` -> `My Account` -> `Create New API Token`
echo '{"username":"MYUSER","key":"MYKEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
kaggle competitions download -c tensorflow2-question-answering
for f in *.zip; do unzip $f; done
rm *.zip
```

# Training / Inference
```bash
# get the code
git clone git@github.com:see--/natural-question-answering.git
# install `transformers`
cd natural-question-answering/transformers_repo
sudo python3 setup.py install
cd ..
# move the data to the root directory
mv ../*.jsonl .
# run the training and evaluation
python3 nq_to_squad.py; python3 train_eval.py
# run inference on the test set
python3 nq_to_squad.py --fn simplified-nq-test.jsonl; python3 train_eval.py --do_not_train --predict_fn nq-test-v1.0.1.json
```

The training and evaluation should finish within 5 hours and you should get a local validation score of ~`0.72` and public LB of ~`0.73`. For inference please check my [Kaggle Notebook](https://www.kaggle.com/seesee/submit-full).

# Notice

Parts of this solution are copied or modified from the following repositories:
* https://github.com/huggingface/transformers
* https://github.com/GoogleCloudPlatform/training-data-analyst/tree/master/courses/fast-and-lean-data-science

Thanks to the authors!