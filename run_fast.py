import subprocess
import time
import os

# === 設定 ===
GAME_DIR = "."  # 假設你的 ml_play.py 跟遊戲檔案(config.py)在同一層
TOTAL_ROUNDS = 20  # 跑 20 場
FPS = 600          # 極速模式
DIFFICULTY = "NORMAL"

print(f"🚀 開始極速收集資料... 目標: {TOTAL_ROUNDS} 場")
print(f"⚡ 速度設定: {FPS} FPS (不顯示畫面)")

for i in range(TOTAL_ROUNDS):
    print(f"Running game {i+1}/{TOTAL_ROUNDS}...")
    
    # 修正重點：參數名稱改成標準的 dash 分隔
    cmd = [
        "python", "-m", "mlgame",
        "-f", str(FPS),
        "--no-display",      # <--- 修正這裡：用減號 "-"
        "-i", "ml_play.py",  # 1P AI
        "-i", "ml_play.py",  # 2P AI
        GAME_DIR,            # 遊戲路徑
        "--difficulty", DIFFICULTY,
        "--game_over_score", "5"
    ]
    
    # 執行指令
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # 如果有錯就印出來
    if process.returncode != 0:
        print("❌ 發生錯誤 (Game Error):")
        print(process.stderr)
        
        # 如果 --no-display 還是錯，嘗試印出幫助訊息看正確參數
        if "unrecognized arguments" in process.stderr:
            print("\n💡 提示：你的 MLGame 版本可能不同。嘗試執行 'python -m mlgame --help' 查看正確參數。")
            # 備案：如果新版 MLGame 改用 --nd，可以試試看把上面改成 "--nd"
        break

print("✅ 所有對戰結束！請檢查 log/ 資料夾是否有 .pickle 檔案。")