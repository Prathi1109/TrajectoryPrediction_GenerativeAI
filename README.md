
## Research Questions

The following are the research questions answered through this work:

1. Is it possible to predict the future states of the other objects in the autonomous driving environment and plan the path of the self-driving vehicle using Active Inference approach?
2. What is the possibility to derive uncertainty-based estimators for prediction and planning in autonomous driving?
3. Is it possible to increase the performance of a Deep Active Inference Model using temporal ensemble?


<!--![Generative AI Model](Figures/GenerativeProcess.png | width = 100)-->


<img src="Figures/GenerativeProcess.png" alt="Alt Text" width="300" height="200">


# Generatve AI Model Architecture

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





