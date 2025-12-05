# in-moment-student-reasoning
Learning the probabilities that students are engaging in mechanistic reasoning during a problem-solving conversation.

Data: 

Our dataset MechTalk consists of 10 student problem-solving conversations consiting of 5 student student groups. The problems are all from a Thermodynamics and Fluid Dynamics course, aech student group was presented with one specialized and one general problem (i.e. recieved by all student groups). 

ML Method:

We utilize the Hierarichal Switching Recurrent Dynamical Model (HSRDM) from Ref.(). The original method uses a fully unsupervised Bayesian variational inference procedure to train a hierarichal (system and entity) time series state-space architecture with recurrent feedback. Model demonstrations were on time-series datasets with agent position/velocity-based features. 
For our task, we incorporate the following: 
-Agent language-based features via EmbeddingGemma encoder embeddings 
-Supervised feedback to guide latent states. 

To model observations (aka entity emissions) at each time step, we assume entity specific gaussian distributions with an autoregressive mean and a full-rank covariance.

To model latent entity transitions at each time step, we assume an entity specific categorical distribution over discrete states with a linear transformation of the distribution at the previous time step and of the outputs of the recurrent feedback from the specific entity only. For each system level state, there exists a different probability transition matrix.   

To model latent system transitions at each time step, we assume one categorical distribution over discrete states with a linear transformation of the states at the previous time step and of the outputs of the recurrent feedback from all entities. 



