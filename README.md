# Mech4Mech
Learning the latent probabilities from conversational data that students are engaging in mechanistic reasoning. 

Paper Data: 

Our dataset MechTalk consists of 10 student problem-solving conversations consiting of 5 student student groups. The problems are all from a Thermodynamics and Fluid Dynamics course, and each student group was presented with one specialized and one general problem (i.e. recieved by all student groups).

Training data from our paper can be found --> data.unsupervised_inference.training 
Test data from our paper can be found --> data.unsupervised_inference.test_new_problem and data.unsupervised_inference.test_seen_problem

ML Method:

We adapt the Hierarichal Switching Recurrent Dynamical Model (HSRDM) from (https://openreview.net/pdf/578aa0777465d1a9ef6af3f5cdccf09a0b4d619c.pdf). The original method uses a fully unsupervised Bayesian variational inference procedure to train a hierarichal (system and entity) time series state-space architecture with recurrent feedback. Model demonstrations were on time-series datasets with agent position/velocity-based features. 
For our task, we incorporate the following: 
-Agent language-based features via EmbeddingGemma encoder embeddings 
-Supervised classifier feedback to guide latent state probabilities towards mechanistic/no mechanistic reasoning states. 

To model observation emissions at each time step, we assume entity specific gaussian distributions with an autoregressive mean and a full-rank covariance.

To model latent entity transitions at each time step, we assume an entity specific categorical distribution over discrete states. The distribution is a linear sum of the probabilistic transitions from the previous state and a function of the outputs of the recurrent feedback from the specific entity only. For each system level state, there exists a different probability transition matrix.   

To model latent system transitions at each time step, we assume a categorical distribution over discrete states. The distribution is a linear sum of the probabilistic transitions from the previous state and a function of the outputs of the recurrent feedback from all entities.

Main Paper Experiments:

To replicate our generalization results on test data given our trained model --> run the src.test_modely.py script 

User Instructions with New Data:


