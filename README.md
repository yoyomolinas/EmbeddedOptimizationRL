# Optimizing Embedded Inference using RL  

Running deep neural nets on embedded devices are costly, therefore running big deep networks can be unfeasible for real time applications. Nevertheless, by taking advantage of the problem domain we propose a reinforcement learning framework that decreases the amount of forward pass executions by a factor of 10. We present our results in comparison to a regression baseline. We observe that RL achieves comparable performance to the baseline and is suitable for this domain because it exploits long term dependencies that have an important role in deciding whether to execute a forward pass or not.

  

## Introduction

  

Pannom, is a real-time embedded system that measures advertisement campaign success in outdoor settings and provides demographic information of passerby’s. In simpler terms, Pannom uses camera footage to detect people that pass by billboards and predict if they were interested in the ad, their age and gender. Pannom uses several deep learning models lined up sequentially, each trained independently, and responsible for different tasks. Currently 3 independent networks are utilized; one for localizing faces, one for localizing facial landmarks, and another for classifying gender and age,. Running 3 deep models on a single embedded chip results in computational overhead, which undermines real-time performance. Nevertheless, we realized that there is room for optimization; intelligent allocation of tasks will improve performance. Currently, Pannom predicts age and gender for every person on every frame and assigns the mean of the predicted labels to each person being tracked. This algorithm consumes a large portion of the computational capacity of the embedded chip. It is more optimal to make predictions until the system is sufficiently confident about the person’s age and gender. By intelligently deciding whether to make a prediction, we intend to decrease the number of predictions, while retaining comparable confidence.

  

Age and gender prediction (AGP) is the most costly component of Pannom. Optimizing its behaviour has a huge impact on overall performance. We make use of Markov Decision Processes to model AGP where agents decide whether or not to execute a forward pass for a person. We experiment with different reward functions and algorithms of RL, to learn a policy that minimizes the amount of predictions while making confident conclusions about a person’s age and gender. Furthermore, to compare our results we construct a baseline. Our baseline is a regression model that attempts to regress the reward function. We use this baseline to compare the performance of our main approach, Q-learning with experience replay. After experimenting with various hyperparameters, such as number of epochs in training, discount factor, and type of policy our results show that Q-learning’s performance is comparable to the regression baseline in most cases, and sometimes better by a small margin.

  

## Problem Definition

  

Age and gender prediction makes up about half of all the computations done by our system. This is due to architectural choices; for AGP we use a 16 layer CNN architecture, also known as VGG-16 [1], infamous for not being suitable for real-time applications. A forward pass is computed in approximately 0.1 seconds for a batch of 10 people by the Nvidia’s Jetson TX2 chip. Hence, in crowded settings, AGP easily becomes the bottleneck of our system. Our current algorithm computes AGP for each person on each frame. If there are 10 people on average for 100 frames, then we would make 1000 predictions. In order to decrease the computational overload, the system needs to predict less. In fact, it is intuitive that one doesn’t need to make hundreds of predictions about a person’s age, but instead can predict a few with high confidence. Thus, we model AGP as a Markov Decision Process, where the agent decides to make a prediction or not. The four components, states, action, reward, and transition of our MDP are detailed as follows. 

  
#### States
A state is defined in terms a feature vector that contains the following information: resolution of the detected face, number of predictions made in current episode, and the extracted facial features from the facial landmark localizer.

First, since AGP is trained on images with 128x128 resolution, the model will have lower accuracy on face images with less resolution than this. To verify this assumption, we plot Figure 3, which indicates that gender confidence scores have a positive correlation with image resolution. In other words, the closer the resolution of the face detected is to the original input dimensions the AGP network was trained on, the more confident the network will be.


![Corr Matrix](https://raw.githubusercontent.com/yoyomolinas/embedded-optimization-rl/master/assets/correlation-table.png)
***Figure 3**: Correlation matrix of confidence scores and image resolution.*

Second, the number of predict actions in current episode serves as an indication of how costly an action would be. This feature is a fundamental part of the cost structure of the reward function utilized.

Third, features from facial landmarks network serve as information on whether the orientation of the person’s head is favorable for age and gender prediction. The AGP network is trained on 200k images of cropped heads, which are mostly pictures of people posing to the camera. Inherently, the database is biased towards the display of the head in a particular orientation where all facial features are visible. Thus, to achieve more reliable, and higher confidence scores it is better to execute forward passes for people with highest visibility of facial landmarks. Nevertheless, we don’t hardcode this, but rather let the agent learn which orientations yield higher confidence from experience. 

All in all, states are described in terms of a 1030 dimensional vector.

![Clusters of feature vectors](https://raw.githubusercontent.com/yoyomolinas/embedded-optimization-rl/master/assets/clsuters-reward.png)
***Figure 4**: Clusters of 1030 dimensional feature vectors in 2D. Dimensionality reduced using tSNE. The colors indicate tracks, and shades of colors indicate the activation of a function for that particular feature vector. Clusters of vectors with similar shade is an indication of better performance for deep neural models.*
#### Actions
There are two actions the system can take, predict and don’t predict. In other words, the decision is based on the question of whether it is worth to make a prediction.

#### Reward
The reward function has two components, cost and reward. These components have different definitions for different versions of reward functions that were experimented with. Nevertheless, the intuition is that the reward function is rewards minus the cost. We try to arrange these components so that favorable actions have positive rewards while unfavorables have negative.

For the linear version of the reward function, cost is trivial, it is set to 0.8. This setting yielded best results because higher cost would pull most of the rewards to the negative end of the spectrum, thus prohibiting most actions. On the other hand the rewards are defined as the average of normalized confidence scores and normalized entropy change. Confidence scores represents the level of confidence about a given person’s age and gender and are the outputs of AGP’s last Softmax layer, while entropy changes represent how has the model’s mean confidence about a particular person improved over time and are computed as the difference between entropy of confidences in previous state and current state. Both confidence and entropy change is modelled as a random variable with unknown distribution, and normalized to have mean 1 and standard deviation 1.

We plot Figure 1 to display the distribution of rewards in the Linear Reward Function and the fluctuation of rewards during an episode. The distribution is centered around 0 which shows that the rewards have an even distribution along the positive and negative ends. The fluctuation plot shows that rewards fluctuate uniformly across the duration of an episode which means that the probability of collecting a positive reward towards the end of an episode is nearly the same as collecting a positive rewards towards the beginning of an episode. This is problematic because it creates a contradiction between the intuition we began optimizing and what our models learn. Our intuition was that rewards should be less towards the end of an episode where the agent has already computed tens of forward passes and has enough data to make a confident conclusion about a person’s age and gender, whereas in this scenario the reward function yields rewards in the same range towards the end of an episode as in the beginning. It is important to note that entropy change was incorporated into the reward function specifically to encode this kind of information, whether a forward pass would cause higher or lower entropy. Nevertheless, entropy change didn’t encode the information that would enforce the behaviour we would like to see from our agent. Thus we resort to simple decay reward function.

![Linear reward](https://raw.githubusercontent.com/yoyomolinas/embedded-optimization-rl/master/assets/linear-reward.png)
***Figure 1**: Visualizing the linear reward function. Right plot shows distribution of rewards while left plot visualizes the function’s behaviour across an episode.*


For the simple decay reward function the cost component is a function of number of forward passes the agent computed for current episode. Starting from an initial value, cost gradually increases as more forward passses are computed. The rate at which cost increases is determined experimentally and satisfies the requirement that an episode accommodates 5 forward passes on average. In other words, cost increases in such a speed that after 5 predictions, the difference between costs and rewards are usually negative. As for the rewards, we exclude entropy change from the linear reward function and keep the confidence scores part the same.

See Figure 2 for the distribution of rewards, which are mostly negative, and fluctuation of rewards across an episode which decays below -1 until the 20th step in average.
Finally, as the reward function we utilize the simple decay version of the reward function.

![Decay reward](https://raw.githubusercontent.com/yoyomolinas/embedded-optimization-rl/master/assets/decay-reward.png)
***Figure 2**: Visualizing the simple decay reward function. Right plot shows distribution of rewards while left plot visualizes the function’s behaviour across an episode.*

#### Transitions
Transitions are deterministic. This MDP consists of a single step, because current state does not depend on previous states. In other words any action taken, whether it is to predict or not predict, yields to a terminal state. Hence we conclude that transitions are deterministic.

## Approach

Our dataset  is comprised of facial features and resolutions of faces from videos and the age gender dataset. We currently have 50k data points. 85% of the data is used for training and the remaining for testing.

First of all, the baseline is a regression model that attempts to regress rewards for given states. Our aim is to achieve better results with RL than pure regression.

  

For the baseline model we train a neural network in a supervised fashion, where actions are taken randomly and loss is computed as the euclidean distance between predicted and actual reward. The neural architecture utilized is a network of 3 fully connected layers stacked on top of each other with 128, 64, and 2 outputs each layer respectively. Finally, the baseline model is trained for 3 epochs.

  

In order to compare performance of algorithms we constrain all algorithms to use the same architecture.

For Reinforcement Learning, our main approach is Q-Learning with experience replay. We observed that experience replay stabilizes results obtained from Q-Learning and thus is a fundamental part of training process. If experience replay is not used, there are huge differences in results between two training sessions. A single epoch of training consists of a single epoch of Q-Learning followed by a single epoch of experience replay with batch size of 16. Batch size is determined experimentally; 16 is sufficient to prevent overfitting. We experiment with various discount factors and policies such as random and epsilon greedy policy. We mainly used discount factors in the lower range such as 0.25 and 0.5, because rewards are immediate in our environment. Agents trained with higher discount factors have failed to learn a good policy because they put higher weight into future rewards.

## Implementation Details

The project is built on Python and uses several open source libraries such as Keras, Numpy, Matplotlib, OpenAI Gym, and OpenCV. Our environment was custom built on top of the OpenAI Gym API. The rendering feature which helps to visualize learning is also custom made using OpenCV.

## Results

We present our results in 3 tables. Each table compares a different configuration of q-learning with the baseline. The settings that we change are namely, number of epochs q-learning is trained for, policy type, discount factor and initial cost. Initial cost is the parameter that determines the initial cost the simple decay reward function uses to penalize predictions.


  

First, Table 1 shows behaviour of q-learning when it is run for different number of epochs. Further, last column is an indication of how important experience replay is. Without experience replay the results do not converge, and results are unstable. For example in this case the model learned that not predict action is always better than predict, thus accumulated reward is 0. This table also shows that there is a dramatic improvement in results once number of epochs increase. In fact, q-learning achieves 3 points better than regression if it is trained for 2 epochs.

|Q-Learning Epochs| Replay Epochs | Mean Accumulated Reward| Baseline|
|--|--|--|--|
| <div align="center"> 1 </div> | <div align="center"> 1 </div> | <div align="center"> -5.5 </div> |<div align="center"> 45.74 </div> |
| <div align="center"> 2 </div> | <div align="center"> 2 </div> | <div align="center"> 48.72 </div> |<div align="center"> 45.74 </div> |
| <div align="center"> 2 </div> | <div align="center"> 0 </div> | <div align="center"> 0 </div> |<div align="center"> 65.43 </div> |

***Table 1**: Comparison of number of epochs q learning is trained for. Discount factor is 0. Policy is random. Initial cost is 0.*<br/><br/>

Second, Table 2 portrays q-learning behaviour when the agent is trained with different discount factors. Best margin against the baseline is achieved when discount factor is 0.5 but, discount factor set to 0.25 also achieves better than the baseline. In this set of experiments the algorithm is trained for a single epoch, the policy is random and the initial cost is 0.4.

|Q-Learning Epochs| Mean Accumulated Reward| Baseline|
|--|--|--|
| <div align="center"> 0 </div> | <div align="center"> 27.15 </div> | <div align="center"> 50.43 </div> |
| <div align="center"> 0.5 </div> | <div align="center"> 49.39 </div> | <div align="center"> 39.48 </div> |
| <div align="center"> 0.25 </div> | <div align="center"> 28.15 </div> | <div align="center"> 27.27 </div> |

***Table 2**: Comparison of discount factors. Number of epochs q-learning is trained for is 1. Policy is random. Initial cost is 0.4.*<br/><br/>

Third, Table 3 shows how q-learning behaves under different policies. Q-learning trained with random policy achieves better than the regression baseline. Epsilon greedy approach fails to compete with random policy, and the faster we decay epsilon, in other words the higher the k value in policy the worst the performance is. Due to time constraints we do not investigate the reasons for why this might happening.

|Policy| Mean Accumulated Reward| Baseline|
|--|--|--|
| <div align="center"> Random </div> | <div align="center"> 59.33 </div> | <div align="center"> 58.29 </div> |
| <div align="center"> E-Greedy (k=0.0004) </div> | <div align="center"> 49.04 </div> | <div align="center"> 58.29 </div> |
| <div align="center"> E-Greedy (k=0.0008) </div> | <div align="center"> 36.92 </div> | <div align="center"> 53.85 </div> |

***Table 3**: Comparison of policies. Number of epochs q-learning is trained for is 3. Discount factor is 0.25. Initial cost is 0.*


## Conclusion

To conclude, we described the RL formulation of the problem, analyzed two reward functions, and presented the results and conclusions about our solution. We saw that for under certain configurations q-learning is preferable over regression, but the q-learning requires much fine tuning of its hyper parameters.

Lastly, it is important to point out that although RL and supervised learning achieve similar results, we are able to reduce the system load by a factor of 10. In other words by incorporating this research into Pannom, we are able to achieve similar accuracy in one tenth of the time.

## References

**[1]** K. Simonyan and A. Zisserman. Very deep convolutional networks for large-scale image recognition. In ICLR, 2015.





