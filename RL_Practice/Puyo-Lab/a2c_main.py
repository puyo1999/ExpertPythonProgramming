## 에이전트를 학습하고 결과를 도시하는 파일
# 필요 패키지 임포트

from a2c_learn import A2Cagent
import gym

def main():
    max_episode_num = 1000  # 최대 에피소드 설정
    env_name = "Pendulum-v0"
    env = gym.make(env_name)
    agent = A2Cagent(env)

    agent.train(max_episode_num)

    agent.plot_result()

if __name__ == "__main__":
    main()
