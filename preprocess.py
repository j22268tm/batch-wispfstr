import os
import glob
import torch
import torchaudio
from tqdm import tqdm
import sys
import warnings
import subprocess
import shutil

# Warning suppression
warnings.filterwarnings("ignore", message=".*torchaudio.backend.common.AudioMetaData.*")
warnings.filterwarnings("ignore", message=".*MPEG_LAYER_III subtype is unknown.*")

# 設定
INPUT_DIR = "inputs"
OUTPUT_DIR = "inputs_processed"
TARGET_EXTS = ["*.mp3", "*.wav", "*.m4a", "*.mp4", "*.mov", "*.mkv", "*.flac"]
SAMPLE_RATE = 48000
FORCE_MONO = True
SPLIT_DURATION_SEC = 60  # 1分ごとに分割

def check_gpu():
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"CUDA is available: {torch.cuda.get_device_name(0)} (VRAM: {vram:.2f} GB)")
        return "cuda"
    else:
        print("CUDA not available. Using CPU.")
        return "cpu"

def load_deepfilternet():
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        model, df_state, _ = init_df()
        return model, df_state, enhance, load_audio, save_audio
    except ImportError as e:
        print(f"Error: Failed to import 'deepfilternet'. {e}")
        sys.exit(1)

def get_audio_duration(file_path):
    try:
        # ffprobeを使って正確な長さを取得
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", file_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return float(result.stdout.strip())
    except Exception:
        # 失敗時はtorchaudioで試行（遅い）
        try:
            info = torchaudio.info(file_path)
            return info.num_frames / info.sample_rate
        except:
            return 0.0

def denoise_audio(audio, model, df_state, enhance_func, device):
    """
    ノイズ除去処理を行う関数。
    DeepFilterNet の内部で numpy 変換が行われるため、入力の audio は CPU 上にある必要があります。
    モデルの推論自体は指定されたデバイス (GPU等) で行われます。
    """
    # モデルをデバイスへ移動
    if device == "cuda":
        model = model.to("cuda")
    else:
        model = model.to("cpu")

    # 音声は CPU のまま渡す (内部で audio.numpy() が呼ばれるため)
    # DeepFilterNet は内部で特徴量をデバイスへ転送して推論を行います
    audio = audio.cpu()
    
    # ノイズ除去実行
    enhanced = enhance_func(model, df_state, audio)
    
    # 結果を確実に CPU へ戻して返す
    return enhanced.detach().cpu()

def process_file_pipeline(file_path, model, df_state, enhance_func, load_audio_func, save_audio_func, device, output_path=None):
    """
    ファイルの読み込み(CPU)、ノイズ除去(GPU/CPU)、保存(CPU)を行うパイプライン。
    """
    try:
        # 1. 音声読み込み (CPU処理)
        # load_audio_func (torchaudio/df) は通常CPUで読み込む
        audio, _ = load_audio_func(file_path, sr=df_state.sr())
        
        # 2. ノイズ除去 (GPU推奨処理)
        try:
            enhanced = denoise_audio(audio, model, df_state, enhance_func, device)
        except RuntimeError as e:
            # VRAM不足時のフォールバック
            if "CUDA out of memory" in str(e) and device == "cuda":
                print(f"  CUDA OOM. Retrying on CPU...")
                torch.cuda.empty_cache()
                # CPUで再試行
                enhanced = denoise_audio(audio, model, df_state, enhance_func, "cpu")
            else:
                raise e
        
        # 元のaudioは不要になったのでCPUへ戻す/解放 (関数スコープ外れれば解放されるが明示的に)
        audio = audio.cpu() 
        
        # 3. 事後処理: モノラル化 (CPU処理)
        # enhancedはdenoise_audioからCPU tensorとして返ってきている
        if FORCE_MONO and enhanced.shape[0] > 1:
            enhanced = torch.mean(enhanced, dim=0, keepdim=True)
            
        # 4. 保存 (CPU処理)
        if output_path:
            save_audio_func(output_path, enhanced, df_state.sr())
        
        return enhanced

    except Exception as e:
        print(f"Error in process_file_pipeline: {e}")
        import traceback
        traceback.print_exc()
        raise e

def process_files():
    device_name = check_gpu()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # モデルロード
    print("Loading DeepFilterNet model...")
    model, df_state, enhance_func, load_audio_func, save_audio_func = load_deepfilternet()

    files = []
    for ext in TARGET_EXTS:
        files.extend(glob.glob(os.path.join(INPUT_DIR, ext)))
    
    if not files:
        print(f"No files found in '{INPUT_DIR}'.")
        return

    print(f"Found {len(files)} files.")

    for file_path in files:
        file_name = os.path.basename(file_path)
        final_output_path = os.path.join(OUTPUT_DIR, os.path.splitext(file_name)[0] + ".wav")
        
        print(f"\nProcessing: {file_name}")
        duration = get_audio_duration(file_path)
        
        if duration > SPLIT_DURATION_SEC:
            print(f"  File is long ({duration/60:.1f} mins). Splitting into chunks...")
            
            # 一時ディレクトリ作成
            temp_dir = os.path.join(OUTPUT_DIR, "temp_split_" + os.path.splitext(file_name)[0])
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            os.makedirs(temp_dir)
            
            try:
                # ffmpegで分割 (wavで出力して劣化を防ぐ)
                split_pattern = os.path.join(temp_dir, "chunk_%03d.wav")
                cmd = [
                    "ffmpeg", "-y", "-i", file_path, 
                    "-f", "segment", 
                    "-segment_time", str(SPLIT_DURATION_SEC), 
                    "-c:a", "pcm_s16le", 
                    split_pattern
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                
                # 分割ファイルを処理
                chunk_files = sorted(glob.glob(os.path.join(temp_dir, "chunk_*.wav")))
                enhanced_chunks_list = os.path.join(temp_dir, "files.txt")
                
                # 出力ディレクトリ作成
                chunk_out_dir = os.path.join(temp_dir, "out")
                os.makedirs(chunk_out_dir, exist_ok=True)

                with open(enhanced_chunks_list, "w") as f:
                    for chunk_file in tqdm(chunk_files, desc="  Chunks"):
                        processed_file = os.path.join(chunk_out_dir, os.path.basename(chunk_file))
                        
                        # Python APIを使用して処理 (GPU/CPU自動選択)
                        process_file_pipeline(
                            chunk_file, model, df_state, enhance_func, 
                            load_audio_func, save_audio_func, device_name, 
                            output_path=processed_file
                        )
                        
                        if not os.path.exists(processed_file):
                            print(f"Error: Processed file not found: {processed_file}")
                            continue

                        # ffmpegのconcatはリストファイルの場所からの相対パスとして解釈する場合があるため
                        # 絶対パスを使うのが最も安全
                        f.write(f"file '{os.path.abspath(processed_file)}'\n")
                
                # 結合
                print("  Merging chunks...")
                cmd_concat = [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
                    "-i", enhanced_chunks_list, 
                    "-c", "copy", 
                    final_output_path
                ]
                # 詳細なエラーを見るためにstderrを表示
                result = subprocess.run(cmd_concat, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
                if result.returncode != 0:
                    print(f"FFmpeg concat failed:\n{result.stderr}")
                    raise subprocess.CalledProcessError(result.returncode, cmd_concat)
                
            except Exception as e:
                print(f"Error processing chunks: {e}")
            finally:
                # お掃除
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
        else:
            # 短いファイルはそのまま処理
            try:
                process_file_pipeline(file_path, model, df_state, enhance_func, load_audio_func, save_audio_func, device_name, output_path=final_output_path)
            except Exception as e:
                print(f"Error: {e}")

    print("-" * 30)
    print(f"Processing complete. Files saved to '{OUTPUT_DIR}/'")

if __name__ == "__main__":
    process_files()
