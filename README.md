# Mech4Mech
An automated tool for learning latent probabilities from conversational data that students are engaging in mechanistic reasoning. 

The associated ArXiv paper "Locating evidence of mechanistic reasoning in student team conversations with mechanistic machine learning" can be found at: 

## **For Users with New Data: Tool Instructions:** 

**Step 1:** Environment Setup

First, clone the repository:

```bash
git clone https://github.com/tufts-ml/Mech4Mech.git
cd Mech4Mech
```
Install [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html). It is a lightweight environment manager that works across platforms. Other tools like `conda` will also work, but commands may need slight adjustments.

Create and activate the environment:

```bash
micromamba env create -f environment.yml
micromamba activate mech4mech
```

Install `ssm` (Required)

The `ssm` package must be installed separately because it does not declare all of its build dependencies.

Run:

```bash
pip install --no-build-isolation git+https://github.com/lindermanlab/ssm.git@eb6c8aa33e5311d3564075807dec340759dd8081
```

**Step 2:** Make an account on HuggingFace and create an access token with READ permissions. Do pip install -U huggingface_hub transformers sentence-transformers -- then copy and paste the acccess token into the login and save to git. This is necessary to use the GemmaEmbedding model as an encoder for the data. 

**Step 3:** Prepare your transcript in a csv format with rows contatining individual student utterances and two columns -- "Student" and "Text". In each row under "Student", there should be a capital letter representing the first initial of the student speaker. Under "Text" is the utterance. It is important that different students are tagged with different first initials. Once prepared, the transcript can be moved into the new_user folder and the new_user.get_user_data.py script can be run. This will generate a user_data.npz file of the data in the same folder that can be used with our trained model. 

**Step 4:** Adapt the src.user_script.py script to run the model on your data. Specifically, (1) change the number of entities J to the total number of students in the trnascript and (2) specify the oututs that you'd like from the model run. 

The user has a choice to generate a csv file with per-student max posterior probabilities (and their assocaited states) across time. The file will be saved as: results/new_user/maxprob.csv

The user also has as choice to plot a segment of the data that has the highest density regions of mechanistic reasoning. The user can specify the the number of time steps in the region. 

Please see the ArXiv paper Sec. V Tool demonstration and user recommendations for more details on considerations prior to tool use. 

## **For Researchers: Replicating ArXiv Paper Results**

**Train/Test data used in ArXiv paper:** 

Our dataset MechTalk consists of 10 student problem-solving conversations consiting of 5 student groups. The problems are all from a Thermodynamics and Fluid Dynamics course at Tufts University. Each student group was presented with one specialized and one general problem (i.e. recieved by all student groups).

Training data from our paper can be found --> data/unsupervised_inference/training 
Test data from our paper can be found --> data/unsupervised_inference/test_new_problem and data/unsupervised_inference/test_seen_problem

**ML Method detailed in RQ1 of ArXiv paper:**

We adapt the Hierarichal Switching Recurrent Dynamical Model (HSRDM) from (https://openreview.net/pdf/578aa0777465d1a9ef6af3f5cdccf09a0b4d619c.pdf). The original method uses a fully unsupervised Bayesian variational inference procedure to train a hierarichal (system and entity) time series state-space architecture with recurrent feedback. Model demonstrations were on time-series datasets with agent position/velocity-based features. 
For our task, we incorporate the following: 
(1) Agent language-based features via EmbeddingGemma encoder embeddings (2) Supervised classifier feedback to guide latent state probabilities towards mechanistic/no mechanistic reasoning states. 

To model observation emissions at each time step, we assume entity specific gaussian distributions with an autoregressive mean and a full-rank covariance.

To model latent entity transitions at each time step, we assume an entity specific categorical distribution over discrete states. The distribution is a linear sum of the probabilistic transitions from the previous state and a function of the outputs of the recurrent feedback from the specific entity only. For each system level state, there exists a different probability transition matrix.   

To model latent system transitions at each time step, we assume a categorical distribution over discrete states. The distribution is a linear sum of the probabilistic transitions from the previous state and a function of the outputs of the recurrent feedback from all entities.

**Main Experiments included in RQ2 of ArXiv paper:**

All trained model parameters for various ablations in Sec. RQ2 can be found in results/unsupervised_inference  

To replicate the generalization results (values for Tables I, II, III) on test data given our trained model --> run the src/test_model.py script. The results will show up in csv files in results/unsupervised_inference_test

To replicate our Figure 3 --> run the src/main_posterior_plot.py script.


