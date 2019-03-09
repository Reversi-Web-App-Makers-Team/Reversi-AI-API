import numpy as np
from hoge import env

class  Brein():
    
     """
     Net(インスタンスを生成)
     update
     """

envs = [env() for i in range(PROCESS_NUM)] 

# envsの初期化

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


class Environment:
    def run(self):
        '''メインの実行'''

        # 同時実行する環境数分、envを生成
        class Environment:
    def run(self):
        '''メインの実行'''

        # 同時実行する環境数分、envを生成
        envs = [env() for i in range(PROCESS_NUM)] 

        # 全エージェントが共有して持つ頭脳Brainを生成
        n_in = envs[0].observation_space.shape[0]  # 状態は64
        n_out = envs[0].action_space.n  # 行動も64
        n_mid = 32
        Net1 = Net(n_in, n_mid, n_out)
        Net2 = Net(n_in, n_mid, n_out) # ディープ・ニューラルネットワークの生成
        global_brain1 = Brain(Net1)
        global_brain2 = Brain(Net2)


        # 格納用変数の生成
        obs_shape = n_in
        current_obs = torch.zeros(
            NUM_PROCESSES, obs_shape)  # torch.Size([16, 64])
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rolloutsのオブジェクト
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 現在の試行の報酬を保持
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 最後の試行の報酬を保持

        obs_np = np.zeros([NUM_PROCESSES, obs_shape])  # Numpy配列
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        each_step = np.zeros(NUM_PROCESSES)  # 各環境のstep数を記録

        episode = 0  # 環境0の試行数

        # 初期状態の開始
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 最新のobsを格納

        # advanced学習用のオブジェクトrolloutsの状態の1つ目に、現在の状態を保存
        rollouts.observations[0].copy_(current_obs)

        # 実行ループ
        for j in range(NUM_EPISODES*NUM_PROCESSES):  # 全体のforループ
            # advanced学習するstep数ごとに計算
            for step in range(NUM_ADVANCED_STEP):

                # 行動を求める
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1)→(16,)→tensorをNumPyに
                actions = action.squeeze(1).numpy()

                # 1stepの実行
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(
                        actions[i])

                    # episodeの終了評価と、state_nextを設定
                    if done_np[i]:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる

                        # 環境0のときのみ出力
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1))
                            episode += 1

                        # 報酬の設定
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 途中でこけたら罰則として報酬-1を与える
                        else:
                            reward_np[i] = 1.0  # 立ったまま終了時は報酬1を与える

                        each_step[i] = 0  # step数のリセット
                        obs_np[i] = envs[i].reset()  # 実行環境のリセット

                    else:
                        reward_np[i] = 0.0  # 普段は報酬0
                        each_step[i] += 1

                # 報酬をtensorに変換し、試行の総報酬に足す
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # 最後の試行の総報酬を更新する
                final_rewards *= masks  # 継続中の場合は1をかけ算してそのまま、done時には0を掛けてリセット
                # 継続中は0を足す、done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                episode_rewards *= masks  # 継続中のmaskは1なのでそのまま、doneの場合は0に

                # 現在の状態をdone時には全部0にする
                current_obs *= masks

                # current_obsを更新
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 最新のobsを格納

                # メモリオブジェクトに今stepのtransitionを挿入
                rollouts.insert(current_obs, action.data, reward, masks)

            # advancedのfor loop終了

            # advancedした最終stepの状態から予想する状態価値を計算

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # rollouts.observationsのサイズはtorch.Size([6, 16, 4])

            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            global_brain.update(rollouts)
            rollouts.after_update()

            # 全部のNUM_PROCESSESが200step経ち続けたら成功
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('連続成功')
                break
        # 全エージェントが共有して持つ頭脳Brainを生成
        n_in = envs[0].observation_space.shape[0]  # 状態は4
        n_out = envs[0].action_space.n  # 行動は2
        n_mid = 32
        actor_critic = Net(n_in, n_mid, n_out)  # ディープ・ニューラルネットワークの生成
        global_brain = Brain(actor_critic)

        # 格納用変数の生成
        obs_shape = n_in
        current_obs = torch.zeros(
            NUM_PROCESSES, obs_shape)  # torch.Size([16, 4])
        rollouts = RolloutStorage(
            NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)  # rolloutsのオブジェクト
        episode_rewards = torch.zeros([NUM_PROCESSES, 1])  # 現在の試行の報酬を保持
        final_rewards = torch.zeros([NUM_PROCESSES, 1])  # 最後の試行の報酬を保持
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])  # Numpy配列
        reward_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        done_np = np.zeros([NUM_PROCESSES, 1])  # Numpy配列
        each_step = np.zeros(NUM_PROCESSES)  # 各環境のstep数を記録
        episode = 0  # 環境0の試行数

        # 初期状態の開始
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()  # torch.Size([16, 4])
        current_obs = obs  # 最新のobsを格納

        # advanced学習用のオブジェクトrolloutsの状態の1つ目に、現在の状態を保存
        rollouts.observations[0].copy_(current_obs)

        # 実行ループ
        for j in range(NUM_EPISODES*NUM_PROCESSES):  # 全体のforループ
            # advanced学習するstep数ごとに計算
            for step in range(NUM_ADVANCED_STEP):

                # 行動を求める
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16,1)→(16,)→tensorをNumPyに
                actions = action.squeeze(1).numpy()

                # 1stepの実行
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(
                        actions[i])

                    # episodeの終了評価と、state_nextを設定
                    if done_np[i]:  # ステップ数が200経過するか、一定角度以上傾くとdoneはtrueになる

                        # 環境0のときのみ出力
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (
                                episode, each_step[i]+1))
                            episode += 1

                        # 報酬の設定
                        if each_step[i] < 195:
                            reward_np[i] = -1.0  # 途中でこけたら罰則として報酬-1を与える
                        else:
                            reward_np[i] = 1.0  # 立ったまま終了時は報酬1を与える

                        each_step[i] = 0  # step数のリセット
                        obs_np[i] = envs[i].reset()  # 実行環境のリセット

                    else:
                        reward_np[i] = 0.0  # 普段は報酬0
                        each_step[i] += 1

                # 報酬をtensorに変換し、試行の総報酬に足す
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 各実行環境それぞれについて、doneならmaskは0に、継続中ならmaskは1にする
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done_np])

                # 最後の試行の総報酬を更新する
                final_rewards *= masks  # 継続中の場合は1をかけ算してそのまま、done時には0を掛けてリセット
                # 継続中は0を足す、done時にはepisode_rewardsを足す
                final_rewards += (1 - masks) * episode_rewards

                # 試行の総報酬を更新する
                episode_rewards *= masks  # 継続中のmaskは1なのでそのまま、doneの場合は0に

                # 現在の状態をdone時には全部0にする
                current_obs *= masks

                # current_obsを更新
                obs = torch.from_numpy(obs_np).float()  # torch.Size([16, 4])
                current_obs = obs  # 最新のobsを格納

                # メモリオブジェクトに今stepのtransitionを挿入
                rollouts.insert(current_obs, action.data, reward, masks)

            # advancedのfor loop終了

            # advancedした最終stepの状態から予想する状態価値を計算

            with torch.no_grad():
                next_value = actor_critic.get_value(
                    rollouts.observations[-1]).detach()
                # rollouts.observationsのサイズはtorch.Size([6, 16, 4])

            # 全stepの割引報酬和を計算して、rolloutsの変数returnsを更新
            rollouts.compute_returns(next_value)

            # ネットワークとrolloutの更新
            global_brain.update(rollouts)
            rollouts.after_update()

            # 全部のNUM_PROCESSESが200step経ち続けたら成功
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('連続成功')
                break
