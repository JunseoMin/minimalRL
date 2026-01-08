# Reinforce review

## Full code
```Python
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import sys
#Hyperparameters
learning_rate = 0.0002
gamma         = 0.98

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []

def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    for n_epi in range(10000):
        s, _ = env.reset()
        print(s)

        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            print(prob)
            print(m)
            print(a)
            sys.exit()
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
    
if __name__ == '__main__':
    main()
```

## Policy Model
```Python
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.data = []
        
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 2)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=0)
        return x
      
    def put_data(self, item):
        self.data.append(item)
        
    def train_net(self):
        R = 0
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []
```
Policy neural network 초기화 코드

```
class Policy(nn.Module):
```
torch mm.Module 상속 (torch default Initializer)
```
self.fc1 = nn.Linear(4, 128)
self.fc2 = nn.Linear(128, 2)
```
policy model architecture 구성
### Forward Process
**input(state)**: 1 \* 4  
s = [$x$, $\dot{x}$, $\theta$, $\dot{\theta}$]  

**output**: 1 \* 2  
[Left prob., Right prob.] (sum == 1)

Forward 할때는 state를 입력으로 받고 정책 확률을 출력해준다.  
입력은 state인데 CartPole에서 카트의 x좌표, x속도, 축 각도, 축 각속도를 의미함.  
출력은 left, right 확률을 제공해줌.

### Train Process
```Python
    def put_data(self, item):
        self.data.append(item)
```
env가 돌려주는 reward와 그때의 행동을 class에 저장하는 부분 ($a_t$,$r_t$) tuple 저장

```Python
    def train_net(self):
        R = 0 #Reward 초기화
        self.optimizer.zero_grad()
        for r, prob in self.data[::-1]:
            R = r + gamma * R # 이전까지의 reward에 gamma적용 + 현재 action에 대한 보상 (0~t)
            loss = -torch.log(prob) * R
            loss.backward()
        self.optimizer.step()
        self.data = []
```

R: 전체 반환값 ($R_t$), $R_0 = 0$, $0\ to\ t$ 의 보상 누적 결과
```
            R = r + gamma * R # 이전 reward에 gamma적용
```

```
            loss = -torch.log(prob) * R
            loss.backward()
```
Objective function 적용 부분  
trajectory에서 현재 state에서 action을 했을때의 loss계산 후 역전파  
prob는 선택한 행동의 확률 (e.g. left move)

## Main
```Python
def main():
    env = gym.make('CartPole-v1')
    pi = Policy()
    score = 0.0
    print_interval = 20
    
    for n_epi in range(10000): # 학습 시작 (10000)최대 10000번 에피소드 진행
        s, _ = env.reset()
        print(s) #[-0.01335475 -0.02720698 -0.03411292  0.00171837]


        done = False
        
        while not done: # CartPole-v1 forced to terminates at 500 step.
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
            print(prob) # tensor([0.4411, 0.5589], grad_fn=<SoftmaxBackward0>)
            print(m) # Categorical(probs: torch.Size([2]))
            print(a) # tensor(0)
            # sys.exit() # for evaluation
            s_prime, r, done, truncated, info = env.step(a.item())
            pi.put_data((r,prob[a]))
            s = s_prime
            score += r
            
        pi.train_net()
        
        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {}".format(n_epi, score/print_interval))
            score = 0.0
    env.close()
```

``` python
            prob = pi(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample()
```
현재 state에서 선택가능한 action들의 확률을 ```prob```변수에 저장한 후 ```Categorical```을 활용해 sampling진행.  
출력으로 0,1(left, right)의 index가 나오고 다음 state로 이동할 수 있게 해줌
```
            s_prime, r, done, truncated, info = env.step(a.item())
```
environment에서 선택된 action을 수행한 후 돌려주는 $s_{t+1}$, reward, 시나리오 실행여부 etc...

```
            pi.put_data((r,prob[a]))
```
Policy network 학습을 위해 trajectory저장

이후```pi.train_net()```을 통해 Policy network update!
