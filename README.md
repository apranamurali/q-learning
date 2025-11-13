# Q Learning Algorithm


## AIM
To develop a Python program to find the optimal policy for the given RL environment using Q-Learning and compare the state values with the Monte Carlo method.

## PROBLEM STATEMENT
Develop a Python program to derive the optimal policy using Q-Learning and compare state values with Monte Carlo method.



## Q LEARNING ALGORITHM
1. Initialize Q-table and hyperparameters.
2. Choose an action using the epsilon-greedy policy and execute the action, observe the next state, reward, and update Q-values and repeat until episode ends.
3. After training, derive the optimal policy from the Q-table.
4. Implement the Monte Carlo method to estimate state values.
5. Compare Q-Learning policy and state values with Monte Carlo results for the given RL environment.

## Q LEARNING FUNCTION
### Name:APARNA.M
### Register Number:212223220008
def q_learning(env,
               gamma=1.0,
               init_alpha=0.5,
               min_alpha=0.01,
               alpha_decay_ratio=0.5,
               init_epsilon=1.0,
               min_epsilon=0.1,
               epsilon_decay_ratio=0.9,
               n_episodes=3000):
  
    nS, nA = env.observation_space.n, env.action_space.n
    pi_track = []

    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)

    select_action=lambda state,Q,epsilon: np.argmax(Q[state]) if np.random.random()>epsilon else np.random.randint(len(Q[state]))

    alphas=decay_schedule(
        init_alpha,min_alpha,
        alpha_decay_ratio,
        n_episodes)
    
    epsilons=decay_schedule(
        init_epsilon,min_epsilon,
        epsilon_decay_ratio,
        n_episodes)
    
    for e in tqdm(range(n_episodes),leave=False):
      state,done=env.reset(),False
      action=select_action(state,Q,epsilons[e])
      while not done:
        action=select_action(state,Q,epsilons[e])
        next_state,reward,done,_=env.step(action)
        td_target=reward+gamma*Q[next_state].max()*(not done)
        td_error=td_target-Q[state][action]
        Q[state][action]=Q[state][action]+alphas[e]*td_error
        state=next_state

      Q_track[e]=Q
      pi_track.append(np.argmax(Q,axis=1))

    V=np.max(Q,axis=1)
    pi=lambda s:{s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]
    
    return Q, V, pi, Q_track, pi_track







## OUTPUT:
<img width="614" height="394" alt="image" src="https://github.com/user-attachments/assets/ca139563-3ef0-4bfe-8ce7-3b77c797a2c8" />






<img width="1092" height="607" alt="image" src="https://github.com/user-attachments/assets/6973e777-435e-481a-8f42-7d732f0eee40" />





<img width="1109" height="447" alt="image" src="https://github.com/user-attachments/assets/9dbc2dc1-756d-4ad8-a85d-1fbb8424372f" />






###State value functions of Monte Carlo method:
<img width="1896" height="926" alt="image" src="https://github.com/user-attachments/assets/fb581ad7-de08-4c7f-936f-a759803caabf" />









###State value functions of Qlearning method:
<img width="1904" height="857" alt="image" src="https://github.com/user-attachments/assets/97bb2a52-8781-4bbf-a74e-2998034f2515" />






## RESULT:
Thus, Q-Learning outperformed Monte Carlo in finding the optimal policy and state values for the RL problem.
