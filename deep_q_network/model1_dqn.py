import numpy as np
from collections import namedtuple

import os

import random
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

GAMMA = 0.99 # 時間割引率
NUM_EPISODES = 1000000 # 最大試行回数
BATCH_SIZE = 32
CAPACITY = 10000 # メモリの容量

NUM_ACTIONS = 64
NUM_STATES = 65

class ReplayMemory:
    '''経験を保存するメモリクラスの定義'''

    def __init__(self, CAPACITY):
        self.capacity = CAPACITY # メモリの最大長さ
        self.memory = [] # 経験を保存する変数
        self. index = 0 # 保存するindexを示す変数

    def push(self, state, action, state_next, reward):
        '''transition = (state, action, next_state, reward)をメモリに保存する'''

        if len(self.memory) < self.capacity:
            self.memory.append(None) # メモリが満タンでないときはたす

            self.memory[self.index] = Transition(state, action, state_next, reward)
            self.index = (self.index + 1) % self.capacity # 保存するindexを一つずらす

    def sample(self, batch_size):
        '''batch_size分だけランダムに保存内容を取り出す'''
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        '''関数lenに対して、現在の変数memoryの長さを返す'''
        return len(self.memory)


class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output

class Brain:
    def __init__(self, num_states, num_actions):
        self.num_actions = num_actions # 行動（盤面のどこに置くかの64を出力）

        # 経験を記憶するメモリオブジェクト
        self.memory = ReplayMemory(CAPACITY)

        # NNを構築
        n_in, n_mid, n_out = num_states, 128, num_actions
        self.main_q_network = Net(n_in, n_mid, n_out)
        self.target_q_network = Net(n_in, n_mid, n_out)

        # 最適化手法の設定
        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=0.0001)

    def replay(self, episode):
        '''Experience Replayでネットワークの結合パラメータを学習'''

        # 1.メモリサイズの確認
        next_state = self.memory.memory[0].next_state
        if len(self.memory) < BATCH_SIZE :
            return

        # 2.ミニバッチの作成
        self.batch, self.state_batch, self.action_batch, self.reward_batch, self.non_final_next_states = self.make_minibatch()

        # 3.教師信号Q(s_t, a_t)を算出
        self.expected_state_action_values = self.get_expected_state_action_values()

        # 4.結合パラメータの更新
        self.update_main_q_network(episode)

    def decide_action(self, state, episode):
        '''行動を決定する'''
        if episode < 30000:
            epsilon = 0.5
        else:
            epsilon = 0.5*(1/((episode-30000)+1))
        
        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval() # 推論モード

            with torch.no_grad():
                # ネットワークの出力の最大値のindexを取り出す
                # .view(1,1): [torch.LongTensor of size 1] → size 1x1
                action = self.main_q_network(state).max(1)[1].view(1, 1)
        else:
            # putable position の行動をランダムに返す
            putable_pos, = np.where(state[0] == 2) # stateの先頭に手番が入っているので-1して盤面のindexにする
            putable_pos += -1
            action = torch.LongTensor([[random.choice(list(putable_pos))]])

        return action

    def make_minibatch(self):
        '''2.ミニバッチの作成'''

        # メモリからミニバッチ分のデータを取り出す（全てNoneのときは
        transitions = self.memory.sample(BATCH_SIZE)

        # (state, action, state_next, reward) xBATCH → (state xBATCH, action xBATCH,..）
        batch = Transition(*zip(*transitions))

        # それぞれについて（例えばstateについて)
        # [torch.FloatTensor of size 1x65] xBATCH → torch.FloatTensor of size BATCH_SIZEx65
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        self.next_state_batch = torch.cat([s if s is not None else torch.Tensor([[np.nan for i in range(NUM_STATES)]]) for s in batch.next_state])
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        '''3.教師信号となるQ(s_t, a_t)値を求める'''

        # 3.1 ネットワークを推論モードに
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 ネットワークが出力したQ(s_t, a_t）を求める
        # 実行したアクションa_tに対応するQ値をgatherで引っ張り出す。
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # 3.3 max{(Q(s_t+1, a)}を求める
        # next_stateがあるかをチェックするインデックスマスク
        non_final_mask = torch.ByteTensor(tuple(map(lambda s:s is not None, self.batch.next_state)))
        # まずは全部0にしておく
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 次の状態での最大Q値の行動a_mをmain_Q_netから求める
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]
        # 次の状態があるものだけにフィルター
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 次の状態があるindexの行動a_mのQ値をtarget_Q_netから算出
        next_state_values[non_final_mask] = self.target_q_network(
                self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 教師となるQ(s_t, a_t)値を、Q学習の式から求める

        # ここで、現在の状態と次の状態の手番が同じか異なるかで場合分けが生じる
        index_diff_value, = np.where(self.state_batch[non_final_mask,0] != self.next_state_batch[non_final_mask,0])
        divided_frag = torch.ones(BATCH_SIZE)
        divided_frag[index_diff_value] = -1 * divided_frag[index_diff_value]

        expected_state_action_values = (self.reward_batch + GAMMA * next_state_values) * divided_frag

        return expected_state_action_values

    def update_main_q_network(self, episode):
        '''4.　結合パラメータの更新'''

        # ネットワークを訓練モードに
        self.main_q_network.train()

        # 損失関数の計算
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))
        if episode % 1000 == 0:
            print('loss : {}'.format(loss.item()))

        # パラメータの更新
        self.optimizer.zero_grad() # 勾配をリセット
        loss.backward()
        self.optimizer.step()

    def update_target_q_network(self):
        '''target Q netの同期'''
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:
    def __init__(self, num_states, num_actions):
        '''課題の状態と行動の数を設定する'''
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self, episode):
        '''Q関数を更新する'''
        self.brain.replay(episode)

    def get_action(self, state, episode):
        '''行動を決定する'''
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, state_next, reward):
        '''memoryオブジェクトにstate, action, state_next, rewardの内容を保存する'''
        self.brain.memory.push(state, action, state_next, reward)

    def update_target_q_function(self):
        ''' Target Q-NetworkをMain Q-Networkと同期'''
        self.brain.update_target_q_network()

class Environment:

    def __init__(self):
        num_states = NUM_STATES
        num_actions = NUM_ACTIONS
        self.agent = Agent(num_states, num_actions)

    def run(self):

        for episode in range(NUM_EPISODES):
            state, player = reset() # 環境の初期化

            state = np.array(state)
            state = np.append(np.array([player]), state)

            state = torch.from_numpy(state).type(torch.FloatTensor) # numpy　→ torch.FloatTensor
            state = torch.unsqueeze(state, 0) # size65 → size 1x65

            while(1):

                action = self.agent.get_action(state, episode) # 行動を求める

                # 行動a_tの実行により、s_{t+1}とdoneフラグを求める
                state_raw = list(np.array(state[0][1:]))
                state_next, player, win_los_frag = step(state_raw, action.item(), player)

                # 終了したとき（勝ち負けが決まったか、反則を犯したとき）
                if win_los_frag == 1:
                    reward = torch.FloatTensor([1.0]) # 勝ったとき

                elif win_los_frag == -1:
                    reward = torch.FloatTensor([-1.0]) # 負けたとき

                elif win_los_frag == 2:
                    reward = torch.FloatTensor([0.0]) # 引き分けのとき

                else:
                    reward = torch.FloatTensor([0.0]) # 普段は報酬0
                    state_next = np.array(state_next)
                    state_next = np.append(np.array([player]), state_next)
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0)

                # メモリに経験を追加
                self.agent.memorize(state, action, state_next, reward)

                # Experience ReplayでQ関数を更新する
                self.agent.update_q_function(episode)

                # 観測の更新
                state = state_next

                # 終了時の処理
                if win_los_frag != 0:
                    if (episode % 2 == 0):
                        self.agent.update_target_q_function()
                    break

        
        #resultsディレクトリを作成
        result_dir = 'results'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)           
        # 終了時モデル保存
        torch.save(self.agent.brain.main_q_network.state_dict(), '/content/results/model2.pt')
