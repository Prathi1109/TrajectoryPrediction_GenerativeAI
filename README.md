
## Research Questions

The following are the research questions answered through this work:

1. Is it possible to predict the future states of the other objects in the autonomous driving environment and plan the path of the self-driving vehicle using Active Inference approach?
2. What is the possibility to derive uncertainty-based estimators for prediction and planning in autonomous driving?
3. Is it possible to increase the performance of a Deep Active Inference Model using temporal ensemble?


<!--![Generative AI Model](Figures/GenerativeProcess.png | width = 100)-->
## Generative AI Model

The generative model is learned by estimating the distributions of hidden states P(s) and the posterior, P(s|o). The probability of the observed data P(o) is called the model evidence which quantifies how good are the predictions of the model. To receive higher values of model evidence P(o), we should choose our model parameters accordingly. Hence, maximizing the model evidence is simply minimising the surprise in terms of active inference . Surprise is mathematically represented as -log P(o). 



<img src="Figures/GenerativeProcess.png" alt="Alt Text" width="300" height="200">


# Generative AI Model Architecture

## Encoder- Decoder Network Architecture
<img src="Figures/Encoder.png" alt="Alt Text" width="400" height="500">

## State Transition 
<img src="Figures/Transition.png" alt="Alt Text" width="300" height="500">

## Increasing Reward during training

<img src="Figures/TrainingReward_PF5.png" alt="Alt Text" width="300" height="200">

## Prediction accuracy over unseen data for 100 episodes

| Feature       | Prediction Accuracy | Reconstruction Accuracy |
| ------------- | ------------------- | ----------------------- |
| x position    | 93.78               | 91.49                   |
| y position    | 95.80.              | 95.87.                  |
| x velocity    | 96.12.              | 95.70                   |
| y velocity    | 91.02               | 91.01                   |





