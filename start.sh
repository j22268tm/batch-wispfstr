#!/bin/bash

# pyenvの設定を読み込む
export PATH="$HOME/.pyenv/bin:$PATH"
eval "$(pyenv init -)"

# 仮想環境の有効化
source venv/bin/activate

# アプリの実行
python app.py
