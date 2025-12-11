# 🏓 乒乓球 AI 模型技術文件 (PingPong AI Technical Documentation)

## 一、數據收集模型 (Data Collection - Physics Expert)

此部分介紹用於收集訓練數據的「專家模型」。該模型採用基於物理規則的暴力演算法，利用直線運動方程式與鏡像反射原理計算球的精確落點，作為「正確答案」。

### 1. 時間預測公式
計算球體移動至目標板子高度 (y_target) 所需的時間 (t)：

> **t = (y_target - y_ball) / v_y**

* **y_target**: 目標高度 (1P 玩家為 420，2P 玩家為 80)
* **y_ball**: 當前球體的 Y 座標
* **v_y**: 球體的垂直速度

### 2. 落點預測 (未考慮牆壁)
計算在不考慮牆壁反彈的情況下，球體將飛行的 X 座標位置 (x_pred)：

> **x_pred = x_ball + v_x * t**

### 3. 牆壁反彈修正公式 (Reflection)
處理球體撞擊牆壁後的實際落點位置。設定場景寬度 W = 200。

* **計算反彈週期 (Cycle):**
  > C = 取整數(x_pred / W)

* **計算餘數位置 (Remainder):**
  > R = x_pred 除以 W 的餘數 (Mod)

* **最終落點判定:**
  * **若 C 為偶數 (0, 2, 4...)**：落點 Target_X = R
  * **若 C 為奇數 (1, 3, 5...)**：落點 Target_X = W - R

---

## 二、深度學習模型 (Deep Learning Model - Multi-Layer Perceptron)

本專案採用 **多層感知機 (MLP)** 架構，屬於前饋神經網路，用於預測球的準確落點。

### 1. 網路架構與節點 (Nodes)

* **輸入層 (Input Layer)**: 4 個節點
    * 特徵：x, y, v_x, v_y (球的座標與速度)
* **隱藏層 1 (Hidden Layer 1)**: 64 個神經元
* **隱藏層 2 (Hidden Layer 2)**: 32 個神經元
* **輸出層 (Output Layer)**: 1 個節點
    * 預測目標：落點 X 座標
<img width="1200" height="1200" alt="Code_Generated_Image (1)" src="https://github.com/user-attachments/assets/8894a6a0-5b7d-4d6f-9b98-22efaad224e8" />

### 2. 資料前處理 (Normalization)

為提升學習效率，將數據縮小到 0~1 之間：

* **Input[0]** = x_ball / 200
* **Input[1]** = y_ball / 500
* **Input[2]** = v_x / 50
* **Input[3]** = v_y / 50

### 3. 神經網路運算公式 (Forward Propagation)

設定輸入為 x，權重為 W，偏差為 b，激活函數使用 ReLU (數值小於0變為0)。

**第一層 (Hidden Layer 1):**
> **h1 = ReLU(x * W1 + b1)**
* x 維度: (1, 4)
* W1 維度: (4, 64)
* b1 維度: (64)
* h1 輸出維度: (1, 64)

**第二層 (Hidden Layer 2):**
> **h2 = ReLU(h1 * W2 + b2)**
* W2 維度: (64, 32)
* b2 維度: (32)
* h2 輸出維度: (1, 32)

**輸出層 (Output Layer):**
> **y_pred = h2 * W3 + b3**
* W3 維度: (32, 1)
* b3 維度: (1)
* y_pred: 最終預測值 (純量)

### 4. 資料還原公式 (Denormalization)

將模型輸出的預測值 (0~1) 還原為實際遊戲座標：

> **Target_X = y_pred * 200**

### 5. 特徵提取與正規化列表

| 特徵名稱 | 符號 | 原始單位 | 正規化公式 | 物理意義 |
| :--- | :---: | :---: | :--- | :--- |
| **球體 X 座標** | x_ball | Pixels (0~200) | x / 200 | 球的水平位置 |
| **球體 Y 座標** | y_ball | Pixels (0~500) | y / 500 | 球的垂直位置 |
| **水平速度** | v_x | Pixels/Frame | v_x / 50 | 球往左右飛的速度 |
| **垂直速度** | v_y | Pixels/Frame | v_y / 50 | 球掉落或上升的速度 |

> **註**：速度除以 50 是為了將數值控制在 -1 到 1 之間，避免數值過小導致學習困難。

### 6. Loss Function
<img width="1000" height="600" alt="Code_Generated_Image" src="https://github.com/user-attachments/assets/776ff6b0-cfd4-4e66-82f5-b512e518cb14" />
