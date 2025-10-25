"""
RAGシステム基底クラス
LangChain、BGE-M3、Unity Catalogとの統合を管理
"""

import os
import json
import pandas as pd
import mlflow
from datetime import datetime
from typing import List, Dict, Any, Optional

# 既存ライブラリ使用（絶対保護対象）
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate


class LLMBase:
    """
    RAGシステムの基底クラス
    共通設定・初期化処理・ユーティリティメソッドを提供
    """
    
    # パス定数（Unity Catalogボリューム）
    VOLUME_PATH = "/Volumes/koiso_databircks_16/b_0048/vol"
    INPUT_DIRECTORY = "/Volumes/koiso_databircks_16/b_0048/vol/input_file"
    EXTRACTED_TEXT_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/extracted_text"
    CHUNKED_TEXT_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/chunked_text"
    FAISS_INDEX_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/faiss_index"
    CONVERSATION_HISTORY_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/conversation_history"
    BUILD_LOGS_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/build_logs"
    
    # BGE-M3モデル設定（Unity Catalogモデルレジストリ）
    BGE_M3_CATALOG = "system"
    BGE_M3_SCHEMA = "ai"
    BGE_M3_MODEL_NAME = "bge_m3"
    BGE_M3_VERSION = "2"
    
    # シークレットスコープ設定
    SECRET_SCOPE = "my-secrets"
    OPENAI_API_KEY_NAME = "openai-api-key"
    
    def __init__(self, dbutils):
        """
        基底クラスの初期化
        
        Args:
            dbutils: Databricksのdbutilsオブジェクト
        """
        self.dbutils = dbutils
        self.llm_client = None
        self.bge_m3_model = None
        self.api_key = None
        
        # 基本ディレクトリの作成
        self.ensure_directories()
        
        print("✅ LLMBase初期化完了")
    
    def setup_llm_client(self) -> bool:
        """
        LangChain ChatOpenAIクライアントの設定
        
        Returns:
            bool: 設定成功フラグ
        """
        try:
            # APIキーの取得
            if not self.load_api_key():
                return False
            
            # ChatOpenAIクライアントの設定
            self.llm_client = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.7,
                max_tokens=2000,
                openai_api_key=self.api_key
            )
            
            print("✅ LLMクライアント設定完了")
            return True
            
        except Exception as e:
            print(f"❌ LLMクライアント設定エラー: {e}")
            return False
    
    def setup_bge_m3_model(self) -> bool:
        """
        BGE-M3モデルの設定（Unity Catalogから）
        
        Returns:
            bool: 設定成功フラグ
        """
        try:
            # モデルURIの構築（正しい形式: models:/catalog.schema.model_name/version）
            model_uri = f"models:/{self.BGE_M3_CATALOG}.{self.BGE_M3_SCHEMA}.{self.BGE_M3_MODEL_NAME}/{self.BGE_M3_VERSION}"
            
            # mlflow pyfuncでのモデル読み込み
            self.bge_m3_model = mlflow.pyfunc.load_model(model_uri)
            
            # モデル動作確認（テストクエリ）
            test_df = pd.DataFrame({"input": ["テスト文章"]})
            test_response = self.bge_m3_model.predict(test_df)
            
            # レスポンス形式の確認
            if not self._validate_bge_response(test_response):
                raise ValueError("BGE-M3レスポンス形式が不正です")
            
            print(f"✅ BGE-M3モデル設定完了: {model_uri}")
            return True
            
        except Exception as e:
            print(f"❌ BGE-M3モデル設定エラー: {e}")
            return False
    
    def extract_embedding_vectors(self, response: Dict[str, Any]) -> List[List[float]]:
        """
        BGE-M3のOpenAI API形式レスポンスからベクトル抽出
        
        Args:
            response: BGE-M3の辞書形式レスポンス
            
        Returns:
            List[List[float]]: 埋め込みベクトルのリスト
            
        Raises:
            ValueError: レスポンス形式が不正な場合
        """
        try:
            # data キーの存在確認
            if 'data' not in response:
                raise ValueError("レスポンスに'data'キーが存在しません")
            
            embeddings = []
            for item in response['data']:
                # embedding キーの存在確認
                if 'embedding' not in item:
                    raise ValueError("埋め込みベクトルが見つかりません")
                
                embedding = item['embedding']
                
                # numpy.ndarrayの場合はリストに変換
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                # ベクトルの妥当性確認
                if not isinstance(embedding, list) or len(embedding) != 1024:
                    raise ValueError(f"不正なベクトル形式: 長さ{len(embedding) if isinstance(embedding, list) else 'unknown'}")
                
                embeddings.append(embedding)
            
            print(f"✅ ベクトル抽出完了: {len(embeddings)}件, 次元: 1024")
            return embeddings
            
        except Exception as e:
            print(f"❌ ベクトル抽出エラー: {e}")
            return []
    
    def ensure_directories(self) -> None:
        """
        全必要ディレクトリの存在確認・作成
        """
        directories = [
            self.VOLUME_PATH,
            self.INPUT_DIRECTORY,
            self.EXTRACTED_TEXT_DIR,
            self.CHUNKED_TEXT_DIR,
            self.FAISS_INDEX_DIR,
            self.CONVERSATION_HISTORY_DIR,
            self.BUILD_LOGS_DIR
        ]
        
        for directory in directories:
            try:
                if not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    print(f"✅ ディレクトリ作成: {directory}")
            except Exception as e:
                print(f"❌ ディレクトリ作成エラー {directory}: {e}")
    
    def load_api_key(self) -> bool:
        """
        シークレットスコープからOpenAI APIキーを取得
        
        Returns:
            bool: 取得成功フラグ
        """
        try:
            self.api_key = self.dbutils.secrets.get(
                scope=self.SECRET_SCOPE,
                key=self.OPENAI_API_KEY_NAME
            )
            
            if not self.api_key:
                raise ValueError("APIキーが空です")
            
            print("✅ APIキー取得完了")
            return True
            
        except Exception as e:
            print(f"❌ APIキー取得エラー: {e}")
            return False
    
    def _validate_bge_response(self, response: Any) -> bool:
        """
        BGE-M3レスポンスの形式確認
        
        Args:
            response: BGE-M3からのレスポンス
            
        Returns:
            bool: 形式が正しい場合True
        """
        try:
            if not isinstance(response, dict):
                return False
            
            required_keys = ['object', 'data', 'usage']
            if not all(key in response for key in required_keys):
                return False
            
            if not isinstance(response['data'], list) or len(response['data']) == 0:
                return False
            
            first_item = response['data'][0]
            if 'embedding' not in first_item:
                return False
            
            return True
            
        except:
            return False
    
    def save_json(self, data: Dict[str, Any], file_path: str) -> bool:
        """
        JSONファイルの保存
        
        Args:
            data: 保存するデータ
            file_path: 保存先パス
            
        Returns:
            bool: 保存成功フラグ
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"❌ JSON保存エラー {file_path}: {e}")
            return False
    
    def load_json(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        JSONファイルの読み込み
        
        Args:
            file_path: ファイルパス
            
        Returns:
            Optional[Dict]: 読み込んだデータ（失敗時はNone）
        """
        try:
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ JSON読み込みエラー {file_path}: {e}")
            return None
    
    def get_file_size_mb(self, file_path: str) -> float:
        """
        ファイルサイズ（MB）の取得
        
        Args:
            file_path: ファイルパス
            
        Returns:
            float: ファイルサイズ（MB）
        """
        try:
            if os.path.exists(file_path):
                size_bytes = os.path.getsize(file_path)
                return size_bytes / (1024 * 1024)
            return 0.0
        except:
            return 0.0