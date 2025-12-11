"""
Step 2: Training Script (Run this offline)
完整修正版 - 包含 2P 鏡像翻轉與正確的正規化公式
"""
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import glob
import os

# === MLP 回歸模型結構 ===
class PingPongRegressor(nn.Module):
    def __init__(self):
        super(PingPongRegressor, self).__init__()
        # Input: 4 (球x, 球y, vx, vy)
        # Output: 1 (預測落點 X)
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# === 準備載入資料 ===
print("Loading data from log/ folder...")
x_data = []
y_data = []

# 讀取 log 資料夾下所有的 pickle 檔
file_list = glob.glob(os.path.join("log", "*.pickle"))

if not file_list:
    print("❌ 錯誤：log 資料夾裡沒有檔案！請先執行 run_fast.py 或手動收集資料")
    exit()

total_samples = 0

for file_path in file_list:
    try:
        with open(file_path, "rb") as f:
            batch_data = pickle.load(f)
            
            # === 自動判斷是否為 2P 數據 ===
            # 如果檔名包含 "2P"，代表這是上面的玩家收集的
            # 我們要把它翻轉成 1P (下面的玩家) 的視角，這樣模型才不會錯亂
            is_2p_data = "2P" in file_path
            
            for d in batch_data:
                # d[0] 是 features: [ball_x, ball_y, vx, vy]
                # d[1] 是 label: [target_x]
                
                bx, by, vx, vy = d[0]
                tx = d[1][0]
                
                # === 鏡像翻轉邏輯 ===
                if is_2p_data:
                    # X 左右相反 (200 - x)
                    # Y 上下顛倒 (500 - y)
                    # 速度方向相反
                    bx = 200 - bx
                    by = 500 - by
                    vx = -vx
                    vy = -vy
                    tx = 200 - tx
                
                # === 正規化 (Normalization) ===
                # 這裡的除數必須跟 AI_ml_play.py 裡的一模一樣！
                features = [
                    bx / 200.0,
                    by / 500.0,
                    vx / 50.0,   # 速度除以 50 (修正版)
                    vy / 50.0    # 速度除以 50 (修正版)
                ]
                
                # Label 也要正規化 (0~1)
                label = [tx / 200.0] 
                
                x_data.append(features)
                y_data.append(label)
                total_samples += 1
                
    except Exception as e:
        print(f"⚠️ Skipping broken file {file_path}: {e}")

print(f"✅ 成功載入 {total_samples} 筆數據！(包含 1P 與 2P 翻轉後的資料)")

if total_samples == 0:
    print("❌ 沒有有效數據，程式結束")
    exit()

# 轉換成 Tensor
x_train = torch.tensor(x_data, dtype=torch.float32)
y_train = torch.tensor(y_data, dtype=torch.float32)

dataset = TensorDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# === 初始化模型與訓練設定 ===
model = PingPongRegressor()
criterion = nn.MSELoss() # 使用均方誤差
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("Start Training...")
epochs = 50 # 訓練 50 輪 (可自行調整)

for epoch in range(epochs): 
    total_loss = 0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    # 每 10 輪印一次進度
    if (epoch+1) % 10 == 0:
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

# === 輸出權重格式 ===
weights = {
    'w1': model.fc1.weight.detach().numpy().T.tolist(),
    'b1': model.fc1.bias.detach().numpy().tolist(),
    'w2': model.fc2.weight.detach().numpy().T.tolist(),
    'b2': model.fc2.bias.detach().numpy().tolist(),
    'w3': model.fc3.weight.detach().numpy().T.tolist(),
    'b3': model.fc3.bias.detach().numpy().tolist()
}

print("\n" + "="*30)
print("✅ 訓練完成！請複製底下的內容到 my_model.py")
print("="*30)
print(f"model = {str(weights)}")
print("="*30)