# reversiDQN

We make machine learning models play Ohello.

#### dqn1, dqn2 :
 - he choose stone putting position where q value is maximized restricted to stone putable position. 
 - deeq q-learning is used to train these agents.
 - dqn2 is stronger than dqn1.
#### sl1:
 - he choose stone putting position where probability of winning the game is maximized restricted to stone putable position.
 - CNN is used. The model has board information as input and predicts where the winner put the stone.
