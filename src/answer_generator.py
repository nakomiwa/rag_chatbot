"""
回答生成クラス
Databricks 16.4 LTS + 既存ライブラリ完全保護版

検索・回答生成に関する具体的な実装を担当
ハイブリッド検索、プロンプト生成、回答生成などを実装
"""

import os
import json
import pickle
from typing import List, Dict, Any, Tuple, Optional
import uuid

# 標準ライブラリ（Databricks標準搭載）
import pandas as pd
import numpy as np

# 既存ライブラリを使用（バージョン変更禁止）
from langchain.memory import ConversationBufferMemory  # 既存: langchain 0.3.21
from langchain_core.prompts import ChatPromptTemplate  # 既存: langchain-core 0.3.51
from langchain_core.documents import Document  # 既存: langchain-core 0.3.51

# 新規インストールライブラリ
import faiss  # faiss-cpu 1.8.0
from rank_bm25 import BM25Okapi  # rank-bm25 0.2.2

# 基底クラスをインポート
from base_llm import LLMBase


class AnswerGenerator(LLMBase):
    """
    回答生成クラス
    
    機能：
    - ハイブリッド検索（ベクトル検索 + BM25テキスト検索）
    - プロンプト生成
    - LLM回答生成
    - 会話履歴管理
    - セッション管理
    """
    
    def __init__(self, dbutils, session_id: str = None, temperature: float = 0.7, 
                 vector_weight: float = 0.7, bm25_weight: float = 0.3, top_k: int = 5):
        """
        回答生成クラスの初期化
        
        Args:
            dbutils: Databricksのユーティリティオブジェクト
            session_id (str, optional): セッションID（未指定時は自動生成）
            temperature (float): LLM温度パラメータ
            vector_weight (float): ベクトル検索の重み
            bm25_weight (float): BM25検索の重み
            top_k (int): 検索結果の上位K件
        """
        super().__init__(dbutils, temperature)
        
        self.session_id = session_id or str(uuid.uuid4())
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.top_k = top_k
        
        # インデックスとドキュメントを読み込み
        self._load_indexes()
        
        # セッション専用メモリを設定
        self.session_memory = self._load_session_memory()
        
        # RAG専用プロンプトテンプレートを設定
        self.rag_prompt = self._setup_rag_prompt()
        
    def _load_indexes(self):
        """
        保存されたFaissとBM25インデックスを読み込み
        """
        try:
            # Faissインデックスを読み込み
            faiss_path = os.path.join(self.FAISS_INDEX_DIR, "faiss.index")
            self.faiss_index = faiss.read_index(faiss_path)
            
            # BM25インデックスを読み込み
            bm25_path = os.path.join(self.FAISS_INDEX_DIR, "bm25_index.pkl")
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
                
            # ドキュメントデータを読み込み
            documents_path = os.path.join(self.FAISS_INDEX_DIR, "documents.json")
            documents_data = self.load_config(documents_path)
            
            self.texts = documents_data["texts"]
            self.metadatas = documents_data["metadatas"]
            
            # インデックス設定を読み込み
            config_path = os.path.join(self.FAISS_INDEX_DIR, "index_config.json")
            self.index_config = self.load_config(config_path)
            
            print(f"インデックス読み込み完了: {len(self.texts)}個のドキュメント")
            
        except Exception as e:
            raise Exception(f"インデックスの読み込みに失敗しました: {str(e)}")
            
    def _load_session_memory(self) -> ConversationBufferMemory:
        """
        セッション専用の会話履歴を読み込み
        
        Returns:
            ConversationBufferMemory: セッション専用メモリ
        """
        session_file = os.path.join(self.CONVERSATION_HISTORY_DIR, f"session_{self.session_id}.json")
        
        # 既存のセッション履歴を読み込み
        session_data = self.load_config(session_file)
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        if session_data and "chat_history" in session_data:
            # 過去の会話履歴を復元
            for entry in session_data["chat_history"]:
                memory.chat_memory.add_user_message(entry["human"])
                memory.chat_memory.add_ai_message(entry["ai"])
                
        return memory
        
    def _setup_rag_prompt(self) -> ChatPromptTemplate:
        """
        RAG専用のプロンプトテンプレートを設定
        
        Returns:
            ChatPromptTemplate: RAG専用プロンプトテンプレート
        """
        return ChatPromptTemplate.from_messages([
            ("system", """あなたは専門的なRAG（検索拡張生成）アシスタントです。
            提供されたコンテキスト情報を基に、正確で有用な回答を生成してください。
            
            回答時の重要な注意点：
            1. 必ずコンテキスト情報に基づいて回答してください
            2. コンテキストに含まれていない情報は推測で答えないでください
            3. 情報が不足している場合は、その旨を明確に伝えてください
            4. 出典ファイル情報を必ず回答の最後に含めてください
            5. 会話履歴を考慮した文脈的な回答を心がけてください
            6. 回答は丁寧で分かりやすい日本語で行ってください
            
            出典情報の表示形式：
            **出典：** [ファイル名] (チャンクID: X)
            """),
            ("human", """以下のコンテキスト情報を基に質問に回答してください。
            
            **検索されたコンテキスト情報：**
            {context}
            
            **会話履歴：**
            {chat_history}
            
            **質問：** {question}
            
            上記の情報を基に、質問に対する正確で有用な回答を生成してください。
            必ず出典ファイル情報も含めて回答してください。""")
        ])
        
    def search_hybrid(self, query: str) -> List[Dict[str, Any]]:
        """
        ハイブリッド検索（ベクトル検索 + BM25検索）を実行
        
        Args:
            query (str): 検索クエリ
            
        Returns:
            List[Dict[str, Any]]: 検索結果のリスト
        """
        # 1. ベクトル検索
        vector_results = self._search_vector(query)
        
        # 2. BM25検索
        bm25_results = self._search_bm25(query)
        
        # 3. スコアを統合してランキング
        hybrid_results = self._combine_search_results(vector_results, bm25_results)
        
        # 4. 上位K件を返却
        return hybrid_results[:self.top_k]
        
    def _search_vector(self, query: str) -> List[Dict[str, Any]]:
        """
        ベクトル検索を実行
        
        Args:
            query (str): 検索クエリ
            
        Returns:
            List[Dict[str, Any]]: ベクトル検索結果
        """
        # クエリの埋め込みベクトルを生成
        query_embedding = self.get_embeddings([query])[0]
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # Faissで類似検索
        scores, indices = self.faiss_index.search(query_vector, min(self.top_k * 2, len(self.texts)))
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.texts):  # 有効なインデックスのみ
                results.append({
                    "index": int(idx),
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "vector_score": float(score),
                    "search_type": "vector"
                })
                
        return results
        
    def _search_bm25(self, query: str) -> List[Dict[str, Any]]:
        """
        BM25検索を実行
        
        Args:
            query (str): 検索クエリ
            
        Returns:
            List[Dict[str, Any]]: BM25検索結果
        """
        # クエリをトークン化
        query_tokens = query.split()
        
        # BM25スコアを計算
        bm25_scores = self.bm25_index.get_scores(query_tokens)
        
        # スコア順にソート
        scored_results = [(score, idx) for idx, score in enumerate(bm25_scores)]
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for score, idx in scored_results[:self.top_k * 2]:
            if score > 0:  # スコアが0より大きいもののみ
                results.append({
                    "index": idx,
                    "text": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "bm25_score": float(score),
                    "search_type": "bm25"
                })
                
        return results
        
    def _combine_search_results(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict[str, Any]]:
        """
        ベクトル検索とBM25検索の結果を統合
        
        Args:
            vector_results (List[Dict]): ベクトル検索結果
            bm25_results (List[Dict]): BM25検索結果
            
        Returns:
            List[Dict[str, Any]]: 統合された検索結果
        """
        # インデックスごとに結果をマージ
        combined_results = {}
        
        # ベクトル検索結果を処理
        for result in vector_results:
            idx = result["index"]
            combined_results[idx] = result.copy()
            combined_results[idx]["vector_score"] = result["vector_score"]
            
        # BM25検索結果を処理
        for result in bm25_results:
            idx = result["index"]
            if idx in combined_results:
                combined_results[idx]["bm25_score"] = result["bm25_score"]
            else:
                combined_results[idx] = result.copy()
                combined_results[idx]["vector_score"] = 0.0
                
        # スコアを正規化して統合
        for result in combined_results.values():
            vector_score = result.get("vector_score", 0.0)
            bm25_score = result.get("bm25_score", 0.0)
            
            # 重み付け統合スコア
            combined_score = (self.vector_weight * vector_score + 
                            self.bm25_weight * bm25_score)
            result["combined_score"] = combined_score
            
        # 統合スコア順にソート
        sorted_results = sorted(combined_results.values(), 
                              key=lambda x: x["combined_score"], reverse=True)
        
        return sorted_results
        
    def generate_answer(self, question: str) -> Dict[str, Any]:
        """
        質問に対する回答を生成
        
        Args:
            question (str): ユーザーの質問
            
        Returns:
            Dict[str, Any]: 回答と関連情報
        """
        try:
            # 1. ハイブリッド検索を実行
            search_results = self.search_hybrid(question)
            
            # 2. コンテキスト情報を構築
            context = self._build_context(search_results)
            
            # 3. 会話履歴を取得
            chat_history = self.session_memory.load_memory_variables({})["chat_history"]
            
            # 4. プロンプトを生成
            prompt = self.rag_prompt.format_messages(
                context=context,
                chat_history=chat_history,
                question=question
            )
            
            # 5. LLMで回答生成
            response = self.llm.invoke(prompt)
            answer = response.content
            
            # 6. 会話履歴を更新
            self.session_memory.save_context(
                {"input": question},
                {"answer": answer}
            )
            
            # 7. セッション履歴を保存
            self._save_session_memory()
            
            # 8. 回答情報を構築
            result = {
                "answer": answer,
                "question": question,
                "search_results": search_results,
                "context": context,
                "session_id": self.session_id,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            return {
                "answer": f"申し訳ございません。回答生成中にエラーが発生しました: {str(e)}",
                "question": question,
                "error": str(e),
                "session_id": self.session_id,
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
    def _build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        検索結果からコンテキスト文字列を構築
        
        Args:
            search_results (List[Dict[str, Any]]): 検索結果
            
        Returns:
            str: 構築されたコンテキスト
        """
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            metadata = result["metadata"]
            source_file = metadata.get("source_file", "不明")
            chunk_id = metadata.get("chunk_id", "不明")
            
            context_part = f"""
【コンテキスト {i}】
出典: {source_file} (チャンクID: {chunk_id})
スコア: {result.get('combined_score', 0.0):.3f}
内容:
{result['text']}
"""
            context_parts.append(context_part)
            
        return "\n".join(context_parts)
        
    def _save_session_memory(self):
        """
        セッションの会話履歴を保存
        """
        try:
            # 会話履歴を取得
            chat_history = self.session_memory.chat_memory.messages
            
            # 保存用データを構築
            session_data = {
                "session_id": self.session_id,
                "chat_history": [],
                "timestamp": pd.Timestamp.now().isoformat()
            }
            
            # メッセージペアを構築
            for i in range(0, len(chat_history), 2):
                if i + 1 < len(chat_history):
                    session_data["chat_history"].append({
                        "human": chat_history[i].content,
                        "ai": chat_history[i + 1].content
                    })
                    
            # ファイルに保存
            session_file = os.path.join(self.CONVERSATION_HISTORY_DIR, f"session_{self.session_id}.json")
            self.save_config(session_data, session_file)
            
        except Exception as e:
            print(f"セッション履歴の保存に失敗しました: {str(e)}")
            
    def reset_session(self):
        """
        セッションをリセット（会話履歴をクリア）
        """
        self.session_memory.clear()
        session_file = os.path.join(self.CONVERSATION_HISTORY_DIR, f"session_{self.session_id}.json")
        if os.path.exists(session_file):
            os.remove(session_file)
        print(f"セッション {self.session_id} をリセットしました")
        
    def get_session_history(self) -> List[Dict[str, str]]:
        """
        セッションの会話履歴を取得
        
        Returns:
            List[Dict[str, str]]: 会話履歴のリスト
        """
        session_file = os.path.join(self.CONVERSATION_HISTORY_DIR, f"session_{self.session_id}.json")
        session_data = self.load_config(session_file)
        
        if session_data and "chat_history" in session_data:
            return session_data["chat_history"]
        else:
            return []