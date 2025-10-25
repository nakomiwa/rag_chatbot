"""
基底LLMクラス
Databricks 16.4 LTS + 既存ライブラリ完全保護版

このクラスは共通処理のみを実装し、具体的な処理は子クラスで実装する設計
既存のlangchain 0.3.21、langchain-core 0.3.51、langchain-text-splitters 0.3.8を活用
"""

import os
import json
from typing import Dict, Any, Optional
import mlflow.pyfunc
import pandas as pd
import numpy as np

# 既存ライブラリを使用（バージョン変更禁止）
from langchain_openai import ChatOpenAI  # 新規インストール: langchain-openai 0.2.9
from langchain.memory import ConversationBufferMemory  # 既存: langchain 0.3.21
from langchain_core.prompts import ChatPromptTemplate  # 既存: langchain-core 0.3.51


class LLMBase:
    """
    LLMの基底クラス
    
    共通処理：
    - Unity Catalogパス定数管理
    - LangChain ChatOpenAIクライアント設定
    - bge_m3モデルのmlflow pyfunc読み込み
    - プロンプトテンプレート基本設定
    - メモリ機能基本設定
    - dbutilsからのAPIキー取得処理
    """
    
    # Unity Catalogボリュームのパス定数
    VOLUME_PATH = "/Volumes/koiso_databircks_16/b_0048/vol/vol"
    INPUT_DIRECTORY = "/Volumes/koiso_databircks_16/b_0048/vol/input_file"
    EXTRACTED_TEXT_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/extracted_text"
    CHUNKED_TEXT_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/chunked_text"
    FAISS_INDEX_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/faiss_index"
    CONVERSATION_HISTORY_DIR = "/Volumes/koiso_databircks_16/b_0048/vol/conversation_history"
    
    def __init__(self, dbutils, temperature: float = 0.7, model_version: str = "2"):
        """
        基底クラスの初期化
        
        Args:
            dbutils: Databricksのユーティリティオブジェクト（外部から受け取り）
            temperature (float): LLM回答生成時のランダム性制御パラメータ（デフォルト: 0.7）
            model_version (str): bge_m3モデルのバージョン（デフォルト: "2"）
        """
        self.dbutils = dbutils
        self.temperature = temperature
        self.model_version = model_version
        
        # 各種ディレクトリの作成
        self._create_directories()
        
        # OpenAI APIキーの取得
        self.api_key = self._get_api_key()
        
        # LangChain ChatOpenAIクライアントの設定
        self.llm = self._setup_llm()
        
        # bge_m3埋め込みモデルの読み込み
        self.embedding_model = self._load_embedding_model()
        
        # メモリ機能の初期化
        self.memory = self._setup_memory()
        
        # プロンプトテンプレートの基本設定
        self.prompt_template = self._setup_prompt_template()
        
    def _create_directories(self):
        """
        必要なディレクトリを作成する
        """
        directories = [
            self.EXTRACTED_TEXT_DIR,
            self.CHUNKED_TEXT_DIR,
            self.FAISS_INDEX_DIR,
            self.CONVERSATION_HISTORY_DIR
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            
    def _get_api_key(self) -> str:
        """
        Databricksシークレットスコープからアクセスキーを取得
        
        Returns:
            str: OpenAI APIキー
            
        Raises:
            Exception: APIキー取得に失敗した場合
        """
        try:
            api_key = self.dbutils.secrets.get(scope="my-secrets", key="openai-api-key")
            if not api_key:
                raise ValueError("APIキーが空です")
            return api_key
        except Exception as e:
            raise Exception(f"APIキーの取得に失敗しました: {str(e)}")
            
    def _setup_llm(self) -> ChatOpenAI:
        """
        LangChain ChatOpenAIクライアントのセットアップ
        
        Returns:
            ChatOpenAI: 設定済みのChatOpenAIインスタンス
        """
        return ChatOpenAI(
            model="gpt-4o-mini",
            api_key=self.api_key,
            temperature=self.temperature,
            max_tokens=4000
        )
        
    def _load_embedding_model(self):
        """
        Unity Catalogからbge_m3モデルをmlflow pyfuncで読み込み
        
        Returns:
            mlflow.pyfunc.PyFuncModel: 読み込み済みのbge_m3モデル
            
        Raises:
            Exception: モデル読み込みに失敗した場合
        """
        try:
            catalog = "system"
            schema = "ai"
            model_name = "bge_m3"
            model_uri = f"models:/{catalog}.{schema}.{model_name}/{self.model_version}"
            
            # mlflow pyfuncでモデルを読み込み
            embedding_model = mlflow.pyfunc.load_model(model_uri)
            
            print(f"bge_m3モデルを読み込みました: {model_uri}")
            return embedding_model
            
        except Exception as e:
            raise Exception(f"bge_m3モデルの読み込みに失敗しました: {str(e)}")
            
    def _setup_memory(self) -> ConversationBufferMemory:
        """
        LangChain ConversationBufferMemoryの基本設定
        
        Returns:
            ConversationBufferMemory: 設定済みのメモリインスタンス
        """
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
    def _setup_prompt_template(self) -> ChatPromptTemplate:
        """
        LangChain ChatPromptTemplateの基本設定
        
        Returns:
            ChatPromptTemplate: 基本的なプロンプトテンプレート
        """
        return ChatPromptTemplate.from_messages([
            ("system", """あなたは専門的なRAG（検索拡張生成）アシスタントです。
            提供されたコンテキスト情報を基に、正確で有用な回答を生成してください。
            
            回答時の注意点：
            1. コンテキスト情報に基づいて回答する
            2. 情報が不足している場合は、その旨を明記する
            3. 出典ファイル情報を必ず含める
            4. 会話履歴を考慮した文脈的な回答を心がける
            """),
            ("human", """コンテキスト情報:
            {context}
            
            会話履歴:
            {chat_history}
            
            質問: {question}
            
            上記の情報を基に、質問に対する回答を生成してください。
            出典ファイル情報も含めて回答してください。""")
        ])
        
    def get_embeddings(self, texts: list) -> list:
        """
        テキストのリストから埋め込みベクトルを生成
        
        Args:
            texts (list): 埋め込み対象のテキストリスト
            
        Returns:
            list: 埋め込みベクトルのリスト
            
        Raises:
            Exception: 埋め込み生成に失敗した場合
        """
        try:
            # pandas DataFrameとして入力を準備
            input_df = pd.DataFrame({"input": texts})
            
            # bge_m3モデルで予測実行
            result = self.embedding_model.predict(input_df)
            
            print(f"bge_m3出力の型: {type(result)}")
            
            # BGE-M3の出力形式に応じた処理
            if isinstance(result, dict):
                print(f"辞書のキー: {list(result.keys())}")
                
                # OpenAI API形式の場合（object, data, usageキー）
                if 'data' in result:
                    data_content = result['data']
                    print(f"dataの型: {type(data_content)}")
                    
                    if isinstance(data_content, list):
                        # dataがリストの場合、各要素から埋め込みベクトルを抽出
                        embeddings = []
                        for i, item in enumerate(data_content):
                            print(f"data[{i}]の型: {type(item)}")
                            if isinstance(item, dict):
                                print(f"data[{i}]のキー: {list(item.keys())}")
                                # 'embedding'キーがある場合
                                if 'embedding' in item:
                                    embeddings.append(item['embedding'])
                                # 'embeddings'キーがある場合
                                elif 'embeddings' in item:
                                    embeddings.append(item['embeddings'])
                                # 数値リストの場合
                                elif isinstance(item, (list, np.ndarray)):
                                    if isinstance(item, np.ndarray):
                                        embeddings.append(item.tolist())
                                    else:
                                        embeddings.append(item)
                                else:
                                    # その他のキーを探す
                                    possible_keys = ['vector', 'values', 'dense_vecs']
                                    found = False
                                    for key in possible_keys:
                                        if key in item:
                                            embedding_value = item[key]
                                            if isinstance(embedding_value, np.ndarray):
                                                embeddings.append(embedding_value.tolist())
                                            else:
                                                embeddings.append(embedding_value)
                                            found = True
                                            break
                                    if not found:
                                        raise ValueError(f"data[{i}]に埋め込みベクトルが見つかりません。キー: {list(item.keys())}")
                            elif isinstance(item, (list, np.ndarray)):
                                # 直接数値配列の場合
                                if isinstance(item, np.ndarray):
                                    embeddings.append(item.tolist())
                                else:
                                    embeddings.append(item)
                            else:
                                raise ValueError(f"data[{i}]が予期しない形式です: {type(item)}")
                                
                    elif isinstance(data_content, np.ndarray):
                        # dataがnumpy配列の場合
                        embeddings = data_content.tolist()
                        
                    else:
                        raise ValueError(f"dataが予期しない形式です: {type(data_content)}")
                        
                # BGE-M3の直接的な出力形式
                elif 'dense_vecs' in result:
                    embeddings_array = result['dense_vecs']
                    if isinstance(embeddings_array, np.ndarray):
                        embeddings = embeddings_array.tolist()
                    else:
                        embeddings = embeddings_array
                        
                elif 'embeddings' in result:
                    embeddings_array = result['embeddings']
                    if isinstance(embeddings_array, np.ndarray):
                        embeddings = embeddings_array.tolist()
                    else:
                        embeddings = embeddings_array
                        
                else:
                    raise ValueError(f"辞書に埋め込みベクトルが見つかりません。利用可能なキー: {list(result.keys())}")
                        
            elif isinstance(result, pd.DataFrame):
                # DataFrameの場合の処理
                if 'dense_vecs' in result.columns:
                    embeddings = result['dense_vecs'].tolist()
                elif len(result.columns) > 0:
                    embeddings = result.iloc[:, 0].tolist()
                else:
                    raise ValueError("DataFrameに埋め込みベクトルが見つかりません")
                    
            elif isinstance(result, np.ndarray):
                # numpy配列の場合
                embeddings = result.tolist()
                
            elif isinstance(result, list):
                # リストの場合
                embeddings = result
                
            else:
                raise ValueError(f"サポートされていない出力形式: {type(result)}")
                
            print(f"処理された埋め込みベクトル数: {len(embeddings)}")
            if len(embeddings) > 0:
                first_embedding = embeddings[0]
                if hasattr(first_embedding, '__len__'):
                    print(f"埋め込みベクトルの次元: {len(first_embedding)}")
                print(f"最初のベクトルのサンプル: {first_embedding[:5] if len(first_embedding) > 5 else first_embedding}")
                
            return embeddings
                
        except Exception as e:
            raise Exception(f"埋め込み生成に失敗しました: {str(e)}")
            
    def save_config(self, config: Dict[str, Any], file_path: str):
        """
        設定情報をJSONファイルに保存
        
        Args:
            config (Dict[str, Any]): 保存する設定情報
            file_path (str): 保存先ファイルパス
        """
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"設定ファイルの保存に失敗しました: {str(e)}")
            
    def load_config(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        JSONファイルから設定情報を読み込み
        
        Args:
            file_path (str): 読み込み元ファイルパス
            
        Returns:
            Optional[Dict[str, Any]]: 読み込んだ設定情報（失敗時はNone）
        """
        try:
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return None
        except Exception as e:
            print(f"設定ファイルの読み込みに失敗しました: {str(e)}")
            return None