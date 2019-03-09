import numpy as np
from hoge import env

class  Brein():
    
     """
     Net(インスタンスを生成)
     update
     """

envs = [env() for i in range(PROCESS_NUM)] 

envsの初期化

for step in range(100):
    if step == 0:
        # 先攻を決める
        player = np.random.choice(-1,1)
    else:
        # 手番を交代
        player *= -1
        
    actions = Net.act(state)
    for board_i in range(PROCESS_NUM):

        states[i], winners[i] = envs[i].step(actions[i])
       
        if winner_flags[i] != 0:
            rewards[i] = winners[i]*player
            envs[i].reset()

        else:
            
            rewards[i] = 0
