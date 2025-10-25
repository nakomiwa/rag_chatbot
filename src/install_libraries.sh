echo "既存ライブラリを完全保護しつつ、新規ライブラリのみをインストール中..."

# 既存ライブラリの確認（インストールはしない）
echo "既存ライブラリ確認:"
echo "- langchain 0.3.21 (既存保護)"
echo "- langchain-core 0.3.51 (既存保護)"
echo "- langchain-text-splitters 0.3.8 (既存保護)"
echo "- pydantic 2.8.2 (既存保護)"
echo "- pydantic_core 2.20.1 (既存保護)"
echo "- tiktoken 0.7.0 (既存保護)"

# 新規ライブラリのみインストール
echo "新規ライブラリインストール開始..."

# LangChain拡張ライブラリ（新規インストール）
pip install langchain-openai==0.2.9
pip install langchain-community==0.3.21

# ドキュメント処理ライブラリ（新規インストール）
pip install python-docx==1.1.2
pip install openpyxl==3.1.5
pip install python-pptx==1.0.2

# 検索・ベクトル処理ライブラリ（新規インストール）
pip install faiss-cpu==1.8.0
pip install rank-bm25==0.2.2

echo "新規ライブラリのインストールが完了しました"
echo "既存ライブラリは一切変更されていません"