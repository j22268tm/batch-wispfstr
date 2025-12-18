import os
import sys
import time
import json
import subprocess
import argparse
import re
import glob
from faster_whisper import WhisperModel

# --- 設定 ---
# テストに使用する音声の長さ（秒）
TEST_DURATION = 15
# 出力する設定ファイル名
CONFIG_FILE = "config.json"
# テスト対象の音声ファイルがあるディレクトリ
INPUT_DIR = "inputs"
# batch.py のパス
BATCH_SCRIPT = "batch.py"

def get_model_size_from_batch():
    """batch.py から MODEL_SIZE を読み取る"""
    try:
        with open(BATCH_SCRIPT, "r", encoding="utf-8") as f:
            content = f.read()
            match = re.search(r'MODEL_SIZE\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "deepdml/faster-whisper-large-v3-turbo-ct2" # デフォルト

MODEL_SIZE = get_model_size_from_batch()

# テスト候補
CANDIDATES = [
    {"device": "cuda", "compute_type": "float16"},
    {"device": "cuda", "compute_type": "int8_float16"},
    {"device": "cuda", "compute_type": "int8"},
    {"device": "cuda", "compute_type": "float32"},
    {"device": "cuda", "compute_type": "bfloat16"},
    
    {"device": "cpu", "compute_type": "int8"},
    {"device": "cpu", "compute_type": "int16"},
    {"device": "cpu", "compute_type": "float32"},
    # {"device": "cpu", "compute_type": "bfloat16"}, # bfloat16 on CPU is rare, can add if needed
]

def get_test_audio_file():
    """テスト用の音声ファイルを探す"""
    exts = ["*.mp3", "*.wav", "*.m4a", "*.mp4", "*.mov", "*.mkv", "*.flac"]
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not files:
        return None
    return files[0]

def run_test_process(device, compute_type, audio_file):
    """
    サブプロセスで単一のテストを実行する
    """
    cmd = [
        sys.executable, 
        __file__, 
        "--worker",
        "--device", device,
        "--compute_type", compute_type,
        "--audio_file", audio_file,
        "--model_size", MODEL_SIZE
    ]
    
    try:
        # タイムアウトを設定してハングアップ防止（60秒）
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=60
        )
        return result
    except subprocess.TimeoutExpired:
        return None

def worker_mode(args):
    """
    実際の推論テストを行うワーカープロセス
    """
    try:
        # モデルロード
        # print(f"Loading model: {args.model_size} on {args.device} ({args.compute_type})...")
        model = WhisperModel(
            args.model_size, 
            device=args.device, 
            compute_type=args.compute_type
        )
        
        # 推論実行 (最初の数秒だけデコード等は難しいので、通常通り実行して時間を測るが、
        # faster-whisperは音声ファイルの一部だけ読み込む機能はないため、
        # decode_audio で読み込んでからカットするなどの工夫が必要だが、
        # ここでは簡易的にファイルそのまま渡す。ただし長すぎると困るので呼び出し元で短いファイルを使う前提が望ましいが
        # 今回は既存ファイルを使うため、transcribeの引数で制御はできない。
        # 代わりに、バイナリで読み込んで BytesIO にしてもいいが、
        # 簡易ベンチマークなので「ロード成功」と「数セグメントの処理」ができればOKとする。
        
        start_time = time.time()
        
        # 実際には全部処理してしまうと長いので、vad_filterなどを使いつつ、
        # ジェネレータを回して最初の1セグメントだけ取得して時間を計測する方法をとる。
        # これにより長いファイルでも即座にベンチマークが終わる。
        segments, info = model.transcribe(
            args.audio_file, 
            beam_size=5,
            vad_filter=True
        )
        
        count = 0
        for _ in segments:
            count += 1
            if count >= 1: # 最初の1セグメント処理できれば動作確認としてはOKとする
                break
        
        duration = time.time() - start_time
        
        # 成功出力 (JSON形式で最後の行に出力する)
        print(json.dumps({"status": "success", "duration": duration, "info_duration": info.duration}))
        sys.exit(0)

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--compute_type")
    parser.add_argument("--audio_file")
    parser.add_argument("--model_size")
    args = parser.parse_args()

    if args.worker:
        worker_mode(args)
        return

    # --- コントローラーモード ---
    print(f"Benchmark Target Model: {MODEL_SIZE}")
    print("-" * 50)

    audio_file = get_test_audio_file()
    if not audio_file:
        print(f"Error: No audio files found in {INPUT_DIR}. Please add a file to test.")
        return

    print(f"Using test file: {audio_file}")
    
    results = []

    for conf in CANDIDATES:
        dev = conf["device"]
        ctype = conf["compute_type"]
        
        print(f"Testing [{dev} / {ctype}] ... ", end="", flush=True) 
        
        proc_result = run_test_process(dev, ctype, audio_file)
        
        if proc_result and proc_result.returncode == 0:
            # 出力の最後の行をパース
            try:
                lines = proc_result.stdout.strip().splitlines()
                last_line = lines[-1]
                data = json.loads(last_line)
                
                duration = data["duration"]
                print(f"OK ({duration:.2f}s)")
                
                results.append({
                    "device": dev,
                    "compute_type": ctype,
                    "duration": duration
                })
            except Exception as e:
                print(f"Failed (Parse Error)")
        else:
            # 失敗時
            err_msg = "Unknown Error"
            if proc_result:
                if proc_result.returncode == -11 or proc_result.returncode == 139:
                    err_msg = "Segmentation Fault (Crash)"
                elif proc_result.stderr:
                    err_msg = proc_result.stderr.strip().split('\n')[-1] # 最後の行だけ表示
            else:
                err_msg = "Timeout"
            
            print(f"Failed: {err_msg[:50]}...")

    print("-" * 50)
    
    if not results:
        print("All configurations failed.")
        return

    # 最速の設定を選択
    best = min(results, key=lambda x: x["duration"])
    print(f"Best Configuration: {best['device']} / {best['compute_type']} ({best['duration']:.2f}s)")

    config_data = {
        "device": best["device"],
        "compute_type": best["compute_type"]
    }

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    
    print(f"Configuration saved to {CONFIG_FILE}")

if __name__ == "__main__":
    main()
