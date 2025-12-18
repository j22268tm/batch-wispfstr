import os
import glob
import time
import sqlite3
from faster_whisper import WhisperModel
import multiprocessing
from tqdm import tqdm

# --- 設定エリア ---
INPUT_DIR = "inputs"
TARGET_EXTS = ["*.mp3", "*.wav", "*.m4a", "*.mp4", "*.mov", "*.mkv", "*.flac"]
MODEL_SIZE = "deepdml/faster-whisper-large-v3-turbo-ct2"

DEVICE = "cpu"
COMPUTE_TYPE = "int8" 
# COMPUTE_TYPE = "float32"

LANGUAGE = "ja"
VAD_FILTER = True

# CPUスレッド設定
# 0なら自動検出(全コア使用)。数値を指定すればその数だけ使う。
# nproc --all が 12 なら 12 などを指定しても良いが、0 (デフォルト) で通常は最大効率になる。
CPU_THREADS = 0 

# DB設定
DB_NAME = "transcription_history.db"
# ------------------

def init_db():
    """データベースの初期化"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            file_name TEXT,
            model_size TEXT,
            device TEXT,
            audio_duration REAL,
            process_time REAL,
            rtf REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def save_history(file_name, model_size, device, duration, process_time):
    """処理結果を保存"""
    if duration <= 0: return
    rtf = process_time / duration
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO history (file_name, model_size, device, audio_duration, process_time, rtf)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (file_name, model_size, device, duration, process_time, rtf))
    conn.commit()
    conn.close()

def get_estimated_rtf(model_size, device):
    """過去のデータから平均RTF (Real Time Factor) を取得"""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        SELECT AVG(rtf) FROM history 
        WHERE model_size = ? AND device = ?
    ''', (model_size, device))
    result = c.fetchone()
    conn.close()
    return result[0] if result and result[0] else None

def format_timestamp(seconds: float):
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def main():
    # DB初期化
    init_db()

    # CPUスレッド数の決定
    max_threads = multiprocessing.cpu_count()
    use_threads = max_threads if CPU_THREADS == 0 else CPU_THREADS
    
    # 環境変数をセットしてバックエンドライブラリに通知
    os.environ["OMP_NUM_THREADS"] = str(use_threads)
    
    print(f"Model: {MODEL_SIZE}")
    print(f"Device: {DEVICE} / Compute Type: {COMPUTE_TYPE}")
    print(f"CPU Threads: {use_threads} (System Max: {max_threads})")
    
    # モデルロード
    # cpu_threads 引数で明示的に並列数を指定
    try:
        model = WhisperModel(
            MODEL_SIZE, 
            device=DEVICE, 
            compute_type=COMPUTE_TYPE, 
            cpu_threads=use_threads
        )
    except Exception as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    # ファイルリスト取得
    files = []
    for ext in TARGET_EXTS:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    files.sort()
    
    if not files:
        print(f"'{INPUT_DIR}' フォルダに処理対象のファイルが見つかりません。")
        return

    print(f"処理対象ファイル数: {len(files)}")
    print("-" * 30)

    for file_path in tqdm(files, desc="Processing Files"):
        file_name = os.path.basename(file_path)
        base_name = os.path.splitext(file_path)[0]
        
        # tqdm.write でログが出力されるようにする（バーが崩れないように）
        tqdm.write(f"処理中: {file_name} ...")
        start_time = time.time()

        try:
            segments, info = model.transcribe(
                file_path, 
                beam_size=5, 
                language=LANGUAGE,
                vad_filter=VAD_FILTER,
                vad_parameters=dict(min_silence_duration_ms=1000) if VAD_FILTER else None
            )

            tqdm.write(f"   -> 言語: {info.language} (確率: {info.language_probability:.2f})")
            tqdm.write(f"   -> 音声の長さ: {info.duration:.2f}秒")

            # 予測時間の計算
            avg_rtf = get_estimated_rtf(MODEL_SIZE, DEVICE)
            if avg_rtf:
                estimated_time = info.duration * avg_rtf
                tqdm.write(f"   -> 予測処理時間: {estimated_time:.1f}秒 (RTF: {avg_rtf:.2f})")
            else:
                tqdm.write(f"   -> 予測処理時間: (データ不足のため計算不可)")

            srt_content = []
            txt_content = []

            # 音声の長さに合わせたプログレスバーを作成
            with tqdm(total=info.duration, unit="s", desc=f"Transcribing {file_name}", leave=False) as pbar:
                for seg_idx, segment in enumerate(segments, 1):
                    start_str = format_timestamp(segment.start)
                    end_str = format_timestamp(segment.end)
                    text = segment.text.strip()
                    
                    # プログレスバーを現在のセグメントの終了位置まで更新
                    pbar.update(segment.end - pbar.n)
                    
                    srt_content.append(f"{seg_idx}\n{start_str} --> {end_str}\n{text}\n")
                    txt_content.append(text)

            # ファイル書き出し
            with open(f"{base_name}.srt", "w", encoding="utf-8") as f:
                f.write("\n".join(srt_content))
            
            with open(f"{base_name}.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(txt_content))

            elapsed = time.time() - start_time
            
            # 実績をDBに保存
            save_history(file_name, MODEL_SIZE, DEVICE, info.duration, elapsed)
            
            tqdm.write(f"   -> 完了 ({elapsed:.1f}秒)")
            tqdm.write(f"   -> 保存: {base_name}.srt / .txt")

        except Exception as e:
            tqdm.write(f"   -> エラー発生: {e}")

    print("-" * 30)
    print("すべての処理が完了しました。")

if __name__ == "__main__":
    main()