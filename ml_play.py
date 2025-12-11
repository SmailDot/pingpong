"""
Step 1: Data Collection (Physics Expert)
"""
import pickle
import numpy as np
import os

class MLPlay:
    def __init__(self, ai_name, *args, **kwargs):
        self.side = ai_name
        print(f"[{ai_name}] Physics Expert (Data Collector) Ready")
        self.data_buffer = []
        self.all_data = []
        self.prev_ball = None
        self.game_exist = False

    def update(self, scene_info, *args, **kwargs):
        if scene_info["status"] != "GAME_ALIVE":
            return "RESET"

        ball = scene_info.get("ball")
        if not ball: return "NONE"
        
        # 取得板子 (相容性寫法)
        platform = scene_info.get("platform")
        if not platform:
             platform = scene_info.get(f"platform_{self.side}")
             if not platform: platform = scene_info.get("platform_1P")

        ball_x, ball_y = ball[0], ball[1]
        
        # 計算球速
        if self.prev_ball is None:
            vx, vy = 0, 0
        else:
            vx = ball_x - self.prev_ball[0]
            vy = ball_y - self.prev_ball[1]
        self.prev_ball = (ball_x, ball_y)

        # === 核心：物理外掛計算落點 (這是我们要 AI 學的答案) ===
        target_x = self.calculate_landing_x(ball_x, ball_y, vx, vy)
        
        # 收集數據：只有當球有速度時才存
        if vy != 0:
            # Input: [球x, 球y, vx, vy]
            # Label: [target_x] (預測的落點)
            self.data_buffer.append([[ball_x, ball_y, vx, vy], [target_x]])

        # === 執行動作 (為了活下去收集更多資料) ===
        plat_x = platform[0]
        plat_center = plat_x + 20
        
        # 簡單的追球邏輯 (置中)
        if plat_center < target_x - 3: return "MOVE_RIGHT"
        elif plat_center > target_x + 3: return "MOVE_LEFT"
        else: return "NONE"

    def calculate_landing_x(self, ball_x, ball_y, vx, vy):
        """ 使用物理公式算出最終落點 """
        scene_width = 200
        # 判定目標高度: 1P 在下(420), 2P 在上(80)
        target_y = 420 if self.side == "1P" else 80

        # 如果球是飛向反方向，暫時預設回到中間 (或預測反彈後的落點，這裡簡化處理)
        if (self.side == "1P" and vy <= 0) or (self.side == "2P" and vy >= 0):
             return 100

        if vy == 0: return 100

        # 計算需要多少 frame 到達板子高度
        steps = (target_y - ball_y) / vy
        
        # 預測原始落點 (未考慮牆壁反彈)
        pred_x = ball_x + vx * steps
        
        # 處理牆壁反彈 (Mirroring)
        # 利用商數和餘數來判斷反彈次數
        # 假設牆壁在 0 和 200
        cycle = int(pred_x // scene_width)
        remain = pred_x % scene_width
        
        # 偶數次反彈：位置不變；奇數次反彈：鏡像位置
        if cycle % 2 == 0:
            return remain
        else:
            return scene_width - remain

    def reset(self):
        print(f"Collecting {len(self.data_buffer)} samples...")
        self.all_data.extend(self.data_buffer)
        self.data_buffer = []
        self.prev_ball = None

        # 只要有資料就存檔，不用等到 2000 筆
        if len(self.all_data) > 0:
            import time
            # 建立一個 log 資料夾
            folder_path = os.path.join(os.path.dirname(__file__), "log")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            
            # 使用時間戳記當檔名，確保不會覆蓋
            # 例如: log/data_1701234567_1P.pickle
            timestamp = int(time.time() * 1000)
            filename = f"data_{timestamp}_{self.side}.pickle"
            filepath = os.path.join(folder_path, filename)
            
            with open(filepath, "wb") as f:
                pickle.dump(self.all_data, f)
            
            print(f"Saved {len(self.all_data)} samples to {filename}")
            self.all_data = []