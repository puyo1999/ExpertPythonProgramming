import numpy as np
import pandas as pd
from torch.utils.data.dataset import Dataset

class Netflix(Dataset):
    def __init__(self):
        # 데이터 읽기
        self.csv = pd.read_csv("CH06.csv")

        # 입력 데이터 정규화
        self.data = self.csv.iloc[:, 1:4].values # 종가를 제외한 데이터
        self.data = self.data / np.max(self.data)

        # 종가 데이터 정규화
        self.label = self.data["Close"].values
        self.label = self.label / np.max(self.label)

    def __len__(self):
        return len(self.data) - 30 # 사용 가능한 배치 개수

    def __getitem__(self, i):
        data = self.data[i:i+30]
        label = self.data[i+30]
        return data, label
