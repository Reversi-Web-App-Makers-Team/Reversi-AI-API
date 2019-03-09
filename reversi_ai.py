import numpy as np

# A2Cのディープ・ニューラルネットワークの構築
import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from hoge import env

class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.actor = nn.Linear(n_mid, n_out)  # 行動を決めるので出力は行動の種類数
        self.critic = nn.Linear(n_mid, 1)  # 状態価値なので出力は1つ

    def forward(self, x):
        '''ネットワークのフォワード計算を定義します'''
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)  # 状態価値の計算
        actor_output = self.actor(h2)  # 行動の計算

        return critic_output, actor_output

    def act(self, x):
        '''状態xから行動を確率的に求めます'''
        value, actor_output = self(x)
        # dim=1で行動の種類方向にsoftmaxを計算
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)  # dim=1で行動の種類方向に確率計算
        return action

    def get_value(self, x):
        '''状態xから状態価値を求めます'''
        value, actor_output = self(x)

        return value

    def evaluate_actions(self, x, actions):
        '''状態xから状態価値、実際の行動actionsのlog確率とエントロピーを求めます'''
        value, actor_output = self(x)

        log_probs = F.log_softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        action_log_probs = log_probs.gather(1, actions)  # 実際の行動のlog_probsを求める

        probs = F.softmax(actor_output, dim=1)  # dim=1で行動の種類方向に計算
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy

class Brain(object):
    def __init__(self, actor_critic):
        self.actor_critic = actor_critic  # actor_criticはクラスNetのディープ・ニューラルネットワーク
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=0.01)

    def update(self, rollouts):
        '''Advantageで計算した5つのstepの全てを使って更新します'''
        obs_shape = rollouts.observations.size()[2:]  # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(
            rollouts.observations[:-1].view(-1, 4),
            rollouts.actions.view(-1, 1))

        # 注意：各変数のサイズ
        # rollouts.observations[:-1].view(-1, 4) torch.Size([80, 4])
        # rollouts.actions.view(-1, 1) torch.Size([80, 1])
        # values torch.Size([80, 1])
        # action_log_probs torch.Size([80, 1])
        # entropy torch.Size([])

        values = values.view(num_steps, num_processes,
                             1)  # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # advantage（行動価値-状態価値）の計算
        advantages = rollouts.returns[:-1] - values  # torch.Size([5, 16, 1])

        # Criticのlossを計算
        value_loss = advantages.pow(2).mean()

        # Actorのgainを計算、あとでマイナスをかけてlossにする
        action_gain = (action_log_probs*advantages.detach()).mean()
        # detachしてadvantagesを定数として扱う

        # 誤差関数の総和
        total_loss = (value_loss * value_loss_coef -
                      action_gain - entropy * entropy_coef)

        # 結合パラメータを更新
        self.actor_critic.train()  # 訓練モードに
        self.optimizer.zero_grad()  # 勾配をリセット
        total_loss.backward()  # バックプロパゲーションを計算
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        #  一気に結合パラメータが変化しすぎないように、勾配の大きさは最大0.5までにする

        self.optimizer.step()  # 結合パラメータを更新

envs = [env() for i in range(PROCESS_NUM)] 

net1 = Net(n_in, n_mid, n_out)
net2 = Net(n_in, n_mid, n_out)

global_brain1 = Brain(net1)
global_brain2 = Brain(net2)

#格納用変数の生成
n_state = n_in
current_state_torch = torch.zeros(
        N_PROCESSES, n_state) # torch.Size([16, 64])

rollouts = RolloutStorage(
        N_ADVANCED_STEP, N_PROCESES, n_state) # ロールアウトインスタンス

episode_reward_torch = torch.zeros([N_PROCESSES, 1])
final_reward_torch = torch.zeros([N_PROCESSES, 1])
state_np = np.zeros([N_PROCESSES, n_state])
reward_np = np.zeros([N_PROCESSES, 1])
winner_np = np.zeros([N_PROCESSES, 1])
each_step_np = np.zeros(N_PROCESSES)
episode = 0 # 試行数

# 初期状態の開始
board_np = np.array([envs[i].reset() for i in range(N_PROCESSES)])
board_torch = torch.from_numpy(board_np).float()
curret_state_torch = board_torch

rollouts.state_torch[0].copy(current_state_torch)

#実行ループ
for i in range(N_EPISODES):
    # advanced学習するstepごとに計算
    for step in range(N_ADVANCED_STEP):

        #行動を求める

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
            each_step[i] = 0

