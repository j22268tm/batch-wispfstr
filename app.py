import gradio as gr
from faster_whisper import WhisperModel
import os
import torch
import gc

# デフォルト設定 (安定性重視でCPU)
# GTX 970等の古いGPUでのクラッシュを避けるため、デフォルトはCPUとする
initial_device = "cpu"
initial_compute_type = "int8"

def format_timestamp(seconds: float):
    """秒数をSRT形式のタイムスタンプ文字列(HH:MM:SS,mmm)に変換"""
    seconds = float(seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

def transcribe(audio, model_key, language, task, beam_size, use_vad, selected_compute_type, device_selection):
    if audio is None:
        return "音声ファイルがありません。", None, None

    # メモリ解放
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # モデルパスのマッピング
    model_map = {
        "large-v3-turbo": "deepdml/faster-whisper-large-v3-turbo-ct2",
        "large-v3": "large-v3",
        "medium": "medium",
        "small": "small",
        "base": "base",
        "tiny": "tiny"
    }
    model_path = model_map.get(model_key, model_key)

    try:
        print(f"Loading model: {model_path}")
        print(f"Running on Device: {device_selection}, Compute Type: {selected_compute_type}")
        
        # モデルのロード
        model = WhisperModel(model_path, device=device_selection, compute_type=selected_compute_type)

        # VADパラメータ
        vad_parameters = dict(min_silence_duration_ms=1000) if use_vad else None

        # 言語設定 ('auto'の場合はNone)
        lang_arg = None if language == "auto" else language

        print("Starting transcription...")
        segments, info = model.transcribe(
            audio, 
            beam_size=beam_size, 
            task=task,
            language=lang_arg,
            vad_filter=use_vad,
            vad_parameters=vad_parameters
        )

        detected_msg = f"Detected language '{info.language}' with probability {info.language_probability:.2f}"
        print(detected_msg)

        # 結果の収集とフォーマット生成
        srt_content = []
        txt_content = []
        display_lines = [detected_msg, "-" * 30]

        # segmentsはジェネレータなのでループで回して処理
        for i, segment in enumerate(segments, start=1):
            start_str = format_timestamp(segment.start)
            end_str = format_timestamp(segment.end)
            text = segment.text.strip()

            # ログ出力
            print(f"[{start_str} --> {end_str}] {text}")

            # SRTフォーマット作成
            srt_content.append(f"{i}\n{start_str} --> {end_str}\n{text}\n")
            
            # TXTフォーマット作成
            txt_content.append(text)
            
            # 画面表示用
            display_lines.append(f"[{start_str}] {text}")

        # 文字列結合
        srt_full_text = "\n".join(srt_content)
        txt_full_text = "\n".join(txt_content)
        display_full_text = "\n".join(display_lines)

        # ファイル出力
        # アップロードされたファイル名ベースで出力ファイル名を作成
        base_name = os.path.splitext(os.path.basename(audio))[0]
        # 一時ファイルとして保存 (gradioの一時ディレクトリまたはcurrent)
        srt_path = os.path.abspath(f"{base_name}.srt")
        txt_path = os.path.abspath(f"{base_name}.txt")

        with open(srt_path, "w", encoding="utf-8") as f:
            f.write(srt_full_text)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(txt_full_text)

        return display_full_text, srt_path, txt_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"エラーが発生しました: {str(e)}\n\n(GTX 970等の古いGPUでは 'cuda' モードは動作しない可能性が高いです。'cpu' モードを推奨します)", None, None

# Gradio UIの構築
with gr.Blocks(title="Faster Whisper WebUI (Turbo)") as demo:
    gr.Markdown("# Faster Whisper WebUI")
    gr.Markdown(f"Recommended Mode: **CPU / int8** (Stable)")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="音声入力")
            
            with gr.Accordion("設定", open=True):
                model_size = gr.Dropdown(
                    choices=["large-v3-turbo", "large-v3", "medium", "small", "base", "tiny"], 
                    value="large-v3-turbo", 
                    label="モデル選択"
                )
                
                # 言語選択
                language = gr.Dropdown(
                    choices=["ja", "en", "zh", "ko", "auto"], 
                    value="ja", 
                    label="言語"
                )

                task = gr.Radio(
                    choices=["transcribe", "translate"], 
                    value="transcribe", 
                    label="タスク"
                )
                
                use_vad = gr.Checkbox(
                    value=True, 
                    label="VADフィルタを有効にする"
                )

                beam_size = gr.Slider(
                    minimum=1, 
                    maximum=10, 
                    value=5, 
                    step=1, 
                    label="Beam Size"
                )

                with gr.Accordion("詳細設定 (CPU推奨)", open=False):
                    # デバイス選択
                    device_selection = gr.Radio(
                        choices=["cpu", "cuda"],
                        value="cpu",
                        label="Device (古いGPUの場合はCPU推奨)"
                    )

                    # Compute Type 選択
                    compute_type_selector = gr.Radio(
                        choices=["int8", "float32", "float16", "int8_float32"],
                        value="int8",
                        label="Compute Type (CPUならint8推奨)"
                    )
            
            btn = gr.Button("文字起こし実行", variant="primary")
        
        with gr.Column(scale=1):
            text_output = gr.Textbox(label="実行ログ / テキストプレビュー", lines=15)
            
            with gr.Row():
                srt_output = gr.File(label="SRTファイル (字幕)")
                txt_output = gr.File(label="TXTファイル (テキスト)")

    btn.click(
        fn=transcribe, 
        inputs=[audio_input, model_size, language, task, beam_size, use_vad, compute_type_selector, device_selection],
        outputs=[text_output, srt_output, txt_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
