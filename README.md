<img width="1341" height="889" alt="image" src="https://github.com/user-attachments/assets/7f8efe0e-7f33-4688-847a-40d792abadde" />


## 一、數據收集模型 (Data Collection - Physics Expert)

此部分介紹用於收集訓練數據的「專家模型」。該模型採用基於物理規則的暴力演算法，利用直線運動方程式與鏡像反射原理計算球的精確落點，作為「正確答案」。

### 1. 時間預測公式
計算球體移動至目標板子高度 (y_target) 所需的時間 (t)：

> **t = (y_target - y_ball) / v_y**

* **y_target**: 目標高度 (1P 玩家為 420，2P 玩家為 70)
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
<img width="1200" height="1200" alt="image" src="https://github.com/user-attachments/assets/402b70c5-7ecd-4857-af0e-abcbde62d2ad" />

### 1.1 訓練資料集

<img width="1511" height="984" alt="image" src="https://github.com/user-attachments/assets/9ca68b0b-329d-4f85-8e9a-465b600c1158" />

- ### 訓練資料集用359個.pickle檔案作為訓練資料用359個.pickle檔案作為訓練資料



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

## 1. 損失函數：均方誤差 (MSE)

我們使用 MSE 來計算模型預測的落點與真實落點之間的差距。

### 🧮 數學公式
$$MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

### 📝 符號說明 (Symbol Legend)

| 符號 (Symbol) | 意義 (Meaning) | 在本專案中的對應 |
| :--- | :--- | :--- |
| **$MSE$** | **均方誤差** (Mean Squared Error) | 最終算出來的「錯誤分數」，越接近 0 代表模型越準。 |
| **$n$** | **樣本總數** (Batch Size) | 每一批次訓練的資料筆數 (例如 `64` 筆)。 |
| **$\sum$** | **總和符號** (Summation) | 代表把這一批次裡所有資料的誤差加總起來。 |
| **$i$** | **索引** (Index) | 代表第幾筆資料 (從第 1 筆算到第 n 筆)。 |
| **$y_i$** | **真實值** (Ground Truth) | 實際上球最後掉落的 X 座標 (Label)。 |
| **$\hat{y}_i$** | **預測值** (Predicted Value) | AI 模型猜測球會掉在哪個 X 座標。 |
| **$(...)^2$** | **平方** (Square) | 將誤差平方，用來消除負號並**懲罰較大的失誤**。 |


<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/8b889c0f-aeda-45dd-9eeb-47c5fee20b15" />

- ### 以上的圖可以確定到第50步的時候損失達到0.000066

# 測試方式
## 確認模型正確,本組採用(演算法vs模型)對打五局)


- ### (1p(MLP演算法) vs 2p(物理公式暴力解)
[https://wwwyoutube.com/watch?v=1HdwYs-FdP0](https://youtu.be/7BVF2nzztVY)
| 場次 (Game) | 原檔標籤 | 時間 (Time) | 最終比分 | 結果 | 備註 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Step 1 | 16:21 | **5 - 2** | **1P 獲勝** | 本場出現過平手 (Draw) 與 2P 得分 |
| **2** | Step 2 | 16:22 | **5 - 0** | **1P 獲勝** | 完封 (Perfect Game) |
| **3** | Step 3 | 16:25 | **5 - 0** | **1P 獲勝** | 完封 (Perfect Game) |
| **4** | Step 4 (A) | 16:26 | **5 - 0** | **1P 獲勝** | 完封 (Perfect Game) |
| **5** | Step 4 (B) | 16:28 | **5 - 0** | **1P 獲勝** | 完封 (Perfect Game) |


### 1P (物理公式暴力解) vs 2P (MLP演算法) 對戰總結表

[https://www.youtube.com/watch?v=_z0315H4C6g](https://youtu.be/YVBCedaBfFg)

| 場次 (Game) | 原檔標籤 | 時間 (Time) | 最終比分 (1P-2P) | 結果 | 備註 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **1** | Step 1 | 16:41 | **1 - 5** | **2P (模型) 獲勝** | MLP演算法開局不穩失 1 分，隨後連勝 |
| **2** | Step 2 | 16:58 | **4 - 5** | **2P (模型) 獲勝** | **激戰**：出現多次平手 (Draw)，雙方勢均力敵 |
| **3** | Step 3 | 16:59 | **0 - 5** | **2P (模型) 獲勝** | **完封** (Perfect Game) |
| **4** | Step 4 | 17:00 | **4 - 5** | **2P (模型) 獲勝** | **激戰**：暴力解算法表現強勢，MLP演算法險勝 |
| **5** | Step 5 | 17:02 | **2 - 5** | **2P (模型) 獲勝** | 暴力解算法取得 2 分，但MLP演算法後期回穩 |
