"""
RAGシステム回答生成クラス
ハイブリッド検索、スコア表示、回答生成を担当
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# 既存ライブラリ使用（絶対保護対象）
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage

from llm_base import LLMBase


class RAGRetriever(LLMBase):
    """
    RAGシステム回答生成クラス
    ハイブリッド検索と常時スコア表示機能を提供
    """
    
    # 検索設定
    DEFAULT_TOP_K = 5
    VECTOR_WEIGHT = 0.7
    TEXT_WEIGHT = 0.3
    
    def __init__(self, dbutils):
        """
        RAG回答生成クラスの初期化
        
        Args:
            dbutils: Databricksのdbutilsオブジェクト
        """
        super().__init__(dbutils)
        
        # インデックス
        self.faiss_index = None
        self.bm25_index = None
        self.documents = []
        self.document_metadata = []
        
        # 会話管理
        self.conversation_memory = None
        self.chat_chain = None
        self.prompt_template = None
        
        # 基本設定の実行
        self.setup_llm_client()
        self.setup_bge_m3_model()
        
        print("✅ RAGRetriever初期化完了")
    
    def initialize_retrieval_system(self) -> bool:
        """
        検索・回答生成システムの初期化
        
        Returns:
            bool: 初期化成功フラグ
        """
        try:
            print("🔄 検索システム初期化中...")
            
            # 1. インデックスの読み込み
            if not self.load_indexes():
                print("❌ インデックス読み込みに失敗しました")
                return False
            
            # 2. 会話メモリの初期化
            self.initialize_conversation_memory()
            
            # 3. プロンプトテンプレートの設定
            self.setup_prompt_template()
            
            print("✅ 検索システム初期化完了")
            return True
            
        except Exception as e:
            print(f"❌ 検索システム初期化エラー: {e}")
            return False
    
    def load_indexes(self) -> bool:
        """
        Faiss・BM25インデックスの読み込み
        
        Returns:
            bool: 読み込み成功フラグ
        """
        try:
            # ファイルパス
            faiss_file = os.path.join(self.FAISS_INDEX_DIR, 'faiss.index')
            bm25_file = os.path.join(self.FAISS_INDEX_DIR, 'bm25_index.pkl')
            documents_file = os.path.join(self.FAISS_INDEX_DIR, 'documents.json')
            
            # ファイル存在確認
            required_files = [faiss_file, bm25_file, documents_file]
            for file_path in required_files:
                if not os.path.exists(file_path):
                    print(f"❌ 必要ファイルが見つかりません: {file_path}")
                    return False
            
            # Faissインデックス読み込み
            self.faiss_index = faiss.read_index(faiss_file)
            print(f"✅ Faissインデックス読み込み完了: {self.faiss_index.ntotal}ベクトル")
            
            # BM25インデックス読み込み
            with open(bm25_file, 'rb') as f:
                self.bm25_index = pickle.load(f)
            print(f"✅ BM25インデックス読み込み完了")
            
            # ドキュメントデータ読み込み
            documents_data = self.load_json(documents_file)
            if not documents_data:
                print("❌ ドキュメントデータ読み込みに失敗しました")
                return False
            
            self.documents = documents_data['chunks']
            self.document_metadata = documents_data['metadata']
            
            print(f"✅ ドキュメントデータ読み込み完了: {len(self.documents)}文書")
            
            # 整合性確認
            if len(self.documents) != self.faiss_index.ntotal:
                print(f"⚠️ 文書数とベクトル数が不一致: {len(self.documents)} vs {self.faiss_index.ntotal}")
            
            return True
            
        except Exception as e:
            print(f"❌ インデックス読み込みエラー: {e}")
            return False
    
    def initialize_conversation_memory(self) -> None:
        """会話メモリの初期化"""
        try:
            self.conversation_memory = ConversationBufferMemory(
                return_messages=True,
                memory_key="chat_history",
                max_token_limit=2000
            )
            print("✅ 会話メモリ初期化完了")
            
        except Exception as e:
            print(f"❌ 会話メモリ初期化エラー: {e}")
    
    def setup_prompt_template(self) -> None:
        """プロンプトテンプレートの設定"""
        try:
            system_prompt = """あなたは優秀なAIアシスタントです。提供されたコンテキスト情報を基に、正確で有用な回答を提供してください。

【回答時の注意事項】
1. 提供されたコンテキスト情報を主な情報源として使用してください
2. コンテキストに記載されていない情報については、一般的な知識で補完可能な範囲で回答してください
3. 不明な点や確信が持てない情報については、その旨を明確に述べてください
4. 回答の最後に、参考にしたソースファイルを明記してください

【コンテキスト情報】
{context}

【会話履歴】
{chat_history}"""

            user_prompt = "質問: {question}"
            
            self.prompt_template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            print("✅ プロンプトテンプレート設定完了")
            
        except Exception as e:
            print(f"❌ プロンプトテンプレート設定エラー: {e}")
    
    def hybrid_search(self, query: str, top_k: int = None) -> Dict[str, Any]:
        """
        ハイブリッド検索の実行（ベクトル検索 + テキスト検索）
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位文書数
            
        Returns:
            Dict[str, Any]: 検索結果とスコア情報
        """
        try:
            if top_k is None:
                top_k = self.DEFAULT_TOP_K
            
            # 1. ベクトル検索
            vector_results = self.vector_search(query, top_k * 2)  # 多めに取得
            
            # 2. テキスト検索
            text_results = self.text_search(query, top_k * 2)  # 多めに取得
            
            # 3. ハイブリッドスコア計算
            hybrid_results = self.calculate_hybrid_scores(vector_results, text_results, top_k)
            
            # 4. 検索結果の構造化
            search_results = {
                'query': query,
                'vector_weight': self.VECTOR_WEIGHT,
                'text_weight': self.TEXT_WEIGHT,
                'top_k': top_k,
                'chunks': hybrid_results,
                'timestamp': datetime.now().isoformat()
            }
            
            return search_results
            
        except Exception as e:
            print(f"❌ ハイブリッド検索エラー: {e}")
            return {'query': query, 'chunks': [], 'error': str(e)}
    
    def vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        ベクトル検索の実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位文書数
            
        Returns:
            List[Dict[str, Any]]: ベクトル検索結果
        """
        try:
            # クエリのベクトル化
            query_df = pd.DataFrame({"input": [query]})
            response = self.bge_m3_model.predict(query_df)
            query_embeddings = self.extract_embedding_vectors(response)
            
            if not query_embeddings:
                return []
            
            query_vector = np.array(query_embeddings[0], dtype=np.float32).reshape(1, -1)
            
            # ベクトルの正規化
            faiss.normalize_L2(query_vector)
            
            # Faiss検索実行
            scores, indices = self.faiss_index.search(query_vector, min(top_k, self.faiss_index.ntotal))
            
            results = []
            for i, (score, doc_idx) in enumerate(zip(scores[0], indices[0])):
                if doc_idx < len(self.documents) and doc_idx >= 0:
                    results.append({
                        'document_index': int(doc_idx),
                        'content': self.documents[doc_idx],
                        'metadata': self.document_metadata[doc_idx],
                        'vector_score': float(score),
                        'rank': i + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ ベクトル検索エラー: {e}")
            return []
    
    def text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        BM25テキスト検索の実行
        
        Args:
            query: 検索クエリ
            top_k: 取得する上位文書数
            
        Returns:
            List[Dict[str, Any]]: テキスト検索結果
        """
        try:
            # クエリのトークナイゼーション（文字レベル）
            query_tokens = []
            current_token = ""
            
            for char in query:
                if char.isalnum() or char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポャュョッ':
                    current_token += char
                else:
                    if current_token:
                        query_tokens.append(current_token)
                        current_token = ""
                    if char.strip():
                        query_tokens.append(char)
            
            if current_token:
                query_tokens.append(current_token)
            
            # 2文字以上のトークンのみを使用
            filtered_tokens = [token for token in query_tokens if len(token) >= 2]
            
            if not filtered_tokens:
                return []
            
            # BM25スコア計算
            doc_scores = self.bm25_index.get_scores(filtered_tokens)
            
            # スコアでソート
            scored_docs = [(i, score) for i, score in enumerate(doc_scores)]
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for rank, (doc_idx, score) in enumerate(scored_docs[:top_k]):
                if doc_idx < len(self.documents) and score > 0:
                    results.append({
                        'document_index': doc_idx,
                        'content': self.documents[doc_idx],
                        'metadata': self.document_metadata[doc_idx],
                        'text_score': float(score),
                        'rank': rank + 1
                    })
            
            return results
            
        except Exception as e:
            print(f"❌ テキスト検索エラー: {e}")
            return []
    
    def calculate_hybrid_scores(self, vector_results: List[Dict], text_results: List[Dict], top_k: int) -> List[Dict[str, Any]]:
        """
        ハイブリッドスコアの計算
        
        Args:
            vector_results: ベクトル検索結果
            text_results: テキスト検索結果
            top_k: 最終的に返す文書数
            
        Returns:
            List[Dict[str, Any]]: ハイブリッドスコアでソートされた結果
        """
        try:
            # 文書IDごとにスコアを集約
            doc_scores = {}
            
            # ベクトル検索スコアの正規化と集約
            if vector_results:
                max_vector_score = max(result['vector_score'] for result in vector_results)
                min_vector_score = min(result['vector_score'] for result in vector_results)
                vector_range = max_vector_score - min_vector_score + 1e-10
                
                for result in vector_results:
                    doc_idx = result['document_index']
                    normalized_score = (result['vector_score'] - min_vector_score) / vector_range
                    
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = {
                            'content': result['content'],
                            'metadata': result['metadata'],
                            'vector_score': 0.0,
                            'text_score': 0.0
                        }
                    
                    doc_scores[doc_idx]['vector_score'] = normalized_score
            
            # テキスト検索スコアの正規化と集約
            if text_results:
                max_text_score = max(result['text_score'] for result in text_results)
                min_text_score = min(result['text_score'] for result in text_results)
                text_range = max_text_score - min_text_score + 1e-10
                
                for result in text_results:
                    doc_idx = result['document_index']
                    normalized_score = (result['text_score'] - min_text_score) / text_range
                    
                    if doc_idx not in doc_scores:
                        doc_scores[doc_idx] = {
                            'content': result['content'],
                            'metadata': result['metadata'],
                            'vector_score': 0.0,
                            'text_score': 0.0
                        }
                    
                    doc_scores[doc_idx]['text_score'] = normalized_score
            
            # ハイブリッドスコア計算
            hybrid_results = []
            for doc_idx, scores in doc_scores.items():
                final_score = (
                    self.VECTOR_WEIGHT * scores['vector_score'] +
                    self.TEXT_WEIGHT * scores['text_score']
                )
                
                # 元のスコアも保持
                original_vector_score = next(
                    (r['vector_score'] for r in vector_results if r['document_index'] == doc_idx),
                    0.0
                )
                original_text_score = next(
                    (r['text_score'] for r in text_results if r['document_index'] == doc_idx),
                    0.0
                )
                
                hybrid_results.append({
                    'document_index': doc_idx,
                    'content': scores['content'],
                    'metadata': scores['metadata'],
                    'vector_score': original_vector_score,
                    'text_score': original_text_score,
                    'final_score': final_score,
                    'source_file': scores['metadata'].get('source_file', 'unknown')
                })
            
            # 最終スコアでソート
            hybrid_results.sort(key=lambda x: x['final_score'], reverse=True)
            
            # 上位top_k件を返す
            return hybrid_results[:top_k]
            
        except Exception as e:
            print(f"❌ ハイブリッドスコア計算エラー: {e}")
            return []
    
    def display_search_scores(self, search_results: Dict[str, Any]) -> None:
        """
        検索結果スコアを毎回表示する標準機能（無効化不可）
        
        Args:
            search_results: 検索結果とスコア情報
        """
        try:
            print("="*60)
            print("🔍 検索結果スコア情報")
            print("="*60)
            print(f"検索クエリ: {search_results.get('query', 'N/A')}")
            print(f"ハイブリッド検索重み - ベクトル: {search_results.get('vector_weight', 0.0):.1f}, テキスト: {search_results.get('text_weight', 0.0):.1f}")
            print(f"取得チャンク数: {len(search_results.get('chunks', []))}")
            print("-"*60)
            
            chunks = search_results.get('chunks', [])
            for i, chunk in enumerate(chunks, 1):
                source_file = chunk.get('source_file', 'unknown')
                vector_score = chunk.get('vector_score', 0.0)
                text_score = chunk.get('text_score', 0.0)
                final_score = chunk.get('final_score', 0.0)
                content = chunk.get('content', '')
                
                # 内容プレビュー（最初の100文字）
                content_preview = content[:100] + "..." if len(content) > 100 else content
                
                print(f"【チャンク {i}】 {source_file}")
                print(f"  ベクトルスコア: {vector_score:.3f}")
                print(f"  テキストスコア: {text_score:.3f}")
                print(f"  最終スコア: {final_score:.3f}")
                print(f"  内容: {content_preview}")
                print("-"*30)
            
            print("="*60)
            
        except Exception as e:
            print(f"❌ スコア表示エラー: {e}")
    
    def generate_response(self, question: str) -> str:
        """
        質問に対する回答生成（スコア表示統合版）
        
        Args:
            question: ユーザーの質問
            
        Returns:
            str: 生成された回答
        """
        try:
            # 1. ハイブリッド検索実行
            search_results = self.hybrid_search(question)
            
            # 2. スコア表示（常時実行）
            self.display_search_scores(search_results)
            
            # 3. コンテキスト情報の構築
            context_info = self._build_context_from_search_results(search_results)
            
            # 4. 会話履歴の取得
            chat_history = self._format_chat_history()
            
            # 5. プロンプトの動的生成
            prompt = self.prompt_template.format(
                context=context_info['context_text'],
                chat_history=chat_history,
                question=question
            )
            
            # 6. LLMによる回答生成
            response = self.llm_client.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
            
            # 7. 出典情報の追加
            sources_info = context_info['sources_info']
            if sources_info:
                answer += f"\n\n【参考資料】\n{sources_info}"
            
            # 8. 会話履歴の更新
            self._update_conversation_memory(question, answer)
            
            return answer
            
        except Exception as e:
            error_msg = f"回答生成中にエラーが発生しました: {e}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def _build_context_from_search_results(self, search_results: Dict[str, Any]) -> Dict[str, str]:
        """
        検索結果からコンテキスト情報を構築
        
        Args:
            search_results: 検索結果
            
        Returns:
            Dict[str, str]: コンテキストテキストと出典情報
        """
        try:
            chunks = search_results.get('chunks', [])
            
            if not chunks:
                return {
                    'context_text': "関連する情報が見つかりませんでした。",
                    'sources_info': ""
                }
            
            # コンテキストテキストの構築
            context_parts = []
            sources = set()
            
            for i, chunk in enumerate(chunks, 1):
                content = chunk.get('content', '')
                source_file = chunk.get('source_file', 'unknown')
                final_score = chunk.get('final_score', 0.0)
                
                # コンテキストに追加
                context_parts.append(f"[参考情報 {i}] {content}")
                sources.add(source_file)
            
            context_text = "\n\n".join(context_parts)
            
            # 出典情報の構築
            sources_info = "・" + "\n・".join(sorted(sources)) if sources else ""
            
            return {
                'context_text': context_text,
                'sources_info': sources_info
            }
            
        except Exception as e:
            print(f"❌ コンテキスト構築エラー: {e}")
            return {
                'context_text': "コンテキスト情報の構築に失敗しました。",
                'sources_info': ""
            }
    
    def _format_chat_history(self) -> str:
        """
        会話履歴のフォーマット
        
        Returns:
            str: フォーマットされた会話履歴
        """
        try:
            if not self.conversation_memory:
                return "（会話履歴なし）"
            
            # メモリから会話履歴を取得
            memory_variables = self.conversation_memory.load_memory_variables({})
            chat_history = memory_variables.get('chat_history', [])
            
            if not chat_history:
                return "（会話履歴なし）"
            
            # 最新の3件のみを表示
            recent_history = chat_history[-6:]  # Human-AI ペアで3件
            
            formatted_history = []
            for message in recent_history:
                if isinstance(message, HumanMessage):
                    formatted_history.append(f"ユーザー: {message.content}")
                elif isinstance(message, AIMessage):
                    formatted_history.append(f"AI: {message.content}")
            
            return "\n".join(formatted_history) if formatted_history else "（会話履歴なし）"
            
        except Exception as e:
            print(f"❌ 会話履歴フォーマットエラー: {e}")
            return "（会話履歴の取得に失敗しました）"
    
    def _update_conversation_memory(self, question: str, answer: str) -> None:
        """
        会話履歴の更新
        
        Args:
            question: ユーザーの質問
            answer: AIの回答
        """
        try:
            if self.conversation_memory:
                # メモリに会話を保存
                self.conversation_memory.save_context(
                    {"input": question},
                    {"output": answer}
                )
                
                # 永続化（セッション管理）
                self._save_conversation_session()
                
        except Exception as e:
            print(f"❌ 会話履歴更新エラー: {e}")
    
    def _save_conversation_session(self) -> None:
        """会話セッションの永続化"""
        try:
            if not self.conversation_memory:
                return
            
            # セッションIDの生成（日付ベース）
            session_id = datetime.now().strftime("%Y%m%d_%H")
            session_file = os.path.join(self.CONVERSATION_HISTORY_DIR, f'session_{session_id}.json')
            
            # メモリ変数の取得
            memory_variables = self.conversation_memory.load_memory_variables({})
            chat_history = memory_variables.get('chat_history', [])
            
            # 保存用データの構築
            session_data = {
                'session_id': session_id,
                'messages': [],
                'last_updated': datetime.now().isoformat()
            }
            
            for message in chat_history:
                if isinstance(message, HumanMessage):
                    session_data['messages'].append({
                        'type': 'human',
                        'content': message.content,
                        'timestamp': datetime.now().isoformat()
                    })
                elif isinstance(message, AIMessage):
                    session_data['messages'].append({
                        'type': 'ai',
                        'content': message.content,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # ファイル保存
            self.save_json(session_data, session_file)
            
        except Exception as e:
            print(f"❌ セッション保存エラー: {e}")
    
    def reset_conversation(self) -> None:
        """会話履歴のリセット"""
        try:
            if self.conversation_memory:
                self.conversation_memory.clear()
                print("✅ 会話履歴をリセットしました")
        except Exception as e:
            print(f"❌ 会話履歴リセットエラー: {e}")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """
        会話の統計情報取得
        
        Returns:
            Dict[str, Any]: 会話統計情報
        """
        try:
            if not self.conversation_memory:
                return {'error': '会話メモリが初期化されていません'}
            
            memory_variables = self.conversation_memory.load_memory_variables({})
            chat_history = memory_variables.get('chat_history', [])
            
            human_messages = [msg for msg in chat_history if isinstance(msg, HumanMessage)]
            ai_messages = [msg for msg in chat_history if isinstance(msg, AIMessage)]
            
            return {
                'total_exchanges': len(human_messages),
                'total_messages': len(chat_history),
                'memory_length': len(str(chat_history)),
                'last_interaction': datetime.now().isoformat() if chat_history else None
            }
            
        except Exception as e:
            return {'error': str(e)}