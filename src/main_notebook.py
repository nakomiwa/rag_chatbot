"""
RAGシステム メインノートブック
Databricks 16.4 LTS + 既存ライブラリ完全保護版

使用方法：
1. RAG構築クラスでシステム構築
2. 回答生成クラスで質問応答
3. 会話履歴管理
"""

import os
import json
from typing import Dict, Any, Optional

# 実装したクラスをインポート
from rag_builder import RAGBuilder
from answer_generator import AnswerGenerator

class RAGSystem:
    """
    RAGシステムの統合クラス
    構築と検索の機能を統合して提供
    """
    
    def __init__(self, dbutils):
        """
        RAGシステムの初期化
        
        Args:
            dbutils: Databricksのユーティリティオブジェクト
        """
        self.dbutils = dbutils
        self.builder = None
        self.generator = None
        
    def build_rag_system(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                        target_directory: str = None) -> Dict[str, Any]:
        """
        RAGシステムを構築
        
        Args:
            chunk_size (int): チャンクサイズ
            chunk_overlap (int): オーバーラップサイズ
            target_directory (str, optional): 処理対象ディレクトリ
            
        Returns:
            Dict[str, Any]: 構築結果の統計情報
        """
        print("=== RAGシステム構築開始 ===")
        
        # RAG構築クラスを初期化
        self.builder = RAGBuilder(
            dbutils=self.dbutils,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # ディレクトリ処理を実行
        build_stats = self.builder.process_directory(target_directory)
        
        print("=== RAGシステム構築完了 ===")
        return build_stats
        
    def initialize_generator(self, session_id: str = None, temperature: float = 0.7,
                           vector_weight: float = 0.7, bm25_weight: float = 0.3, 
                           top_k: int = 5) -> str:
        """
        回答生成器を初期化
        
        Args:
            session_id (str, optional): セッションID
            temperature (float): LLM温度パラメータ
            vector_weight (float): ベクトル検索の重み
            bm25_weight (float): BM25検索の重み
            top_k (int): 検索結果の上位K件
            
        Returns:
            str: セッションID
        """
        self.generator = AnswerGenerator(
            dbutils=self.dbutils,
            session_id=session_id,
            temperature=temperature,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
            top_k=top_k
        )
        
        print(f"回答生成器を初期化しました (セッションID: {self.generator.session_id})")
        return self.generator.session_id
        
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        質問に対する回答を生成
        
        Args:
            question (str): ユーザーの質問
            
        Returns:
            Dict[str, Any]: 回答と関連情報
        """
        if not self.generator:
            raise Exception("回答生成器が初期化されていません。initialize_generator()を先に実行してください。")
            
        return self.generator.generate_answer(question)
        
    def search_documents(self, query: str) -> list:
        """
        ドキュメント検索のみを実行（回答生成なし）
        
        Args:
            query (str): 検索クエリ
            
        Returns:
            list: 検索結果
        """
        if not self.generator:
            raise Exception("回答生成器が初期化されていません。initialize_generator()を先に実行してください。")
            
        return self.generator.search_hybrid(query)
        
    def get_conversation_history(self) -> list:
        """
        会話履歴を取得
        
        Returns:
            list: 会話履歴
        """
        if not self.generator:
            return []
            
        return self.generator.get_session_history()
        
    def reset_conversation(self):
        """
        会話履歴をリセット
        """
        if self.generator:
            self.generator.reset_session()
            
    def get_system_stats(self) -> Dict[str, Any]:
        """
        システムの統計情報を取得
        
        Returns:
            Dict[str, Any]: システム統計情報
        """
        stats = {}
        
        # 構築統計情報
        build_stats_path = "/Volumes/b_0048/vol/build_stats.json"
        if os.path.exists(build_stats_path):
            with open(build_stats_path, 'r', encoding='utf-8') as f:
                stats["build_stats"] = json.load(f)
                
        # インデックス統計情報
        index_config_path = "/Volumes/b_0048/vol/faiss_index/index_config.json"
        if os.path.exists(index_config_path):
            with open(index_config_path, 'r', encoding='utf-8') as f:
                stats["index_config"] = json.load(f)
                
        # セッション情報
        if self.generator:
            stats["current_session"] = {
                "session_id": self.generator.session_id,
                "vector_weight": self.generator.vector_weight,
                "bm25_weight": self.generator.bm25_weight,
                "top_k": self.generator.top_k
            }
            
        return stats


# ===============================================
# 使用例とテスト関数
# ===============================================

def example_usage():
    """
    RAGシステムの使用例
    """
    # Databricksのdbutilsオブジェクトを取得
    # 注意: 実際のDatabricks環境では自動的にdbutilsが利用可能
    
    # RAGシステムを初期化
    rag_system = RAGSystem(dbutils)
    
    # 1. RAGシステムを構築（初回のみ実行）
    print("1. RAGシステム構築...")
    build_stats = rag_system.build_rag_system(
        chunk_size=1000,
        chunk_overlap=200
        # target_directory は未指定の場合、デフォルトの /Volumes/b_0048/vol/input_file が使用される
    )
    print(f"構築完了: {build_stats['indexing']['total_documents']}個のドキュメントを処理")
    
    # 2. 回答生成器を初期化
    print("\n2. 回答生成器初期化...")
    session_id = rag_system.initialize_generator(
        temperature=0.7,
        vector_weight=0.7,
        bm25_weight=0.3,
        top_k=5
    )
    
    # 3. 質問応答のテスト
    print("\n3. 質問応答テスト...")
    
    test_questions = [
        "この資料の主な内容は何ですか？",
        "売上に関する情報を教えてください",
        "今後の計画について詳しく説明してください"
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n--- 質問 {i} ---")
        print(f"Q: {question}")
        
        # 回答生成
        result = rag_system.ask_question(question)
        print(f"A: {result['answer']}")
        
        # 検索結果の詳細（必要に応じて）
        if len(result.get('search_results', [])) > 0:
            print(f"検索ヒット数: {len(result['search_results'])}件")
            
    # 4. 会話履歴の確認
    print("\n4. 会話履歴確認...")
    history = rag_system.get_conversation_history()
    print(f"会話回数: {len(history)}回")
    
    # 5. システム統計情報の表示
    print("\n5. システム統計情報...")
    stats = rag_system.get_system_stats()
    if "index_config" in stats:
        print(f"インデックス済みドキュメント数: {stats['index_config']['total_documents']}")
        print(f"ベクトル次元数: {stats['index_config']['faiss_dimension']}")


def simple_qa_interface():
    """
    簡単な質問応答インターフェース
    """
    # RAGシステムを初期化
    rag_system = RAGSystem(dbutils)
    
    # 回答生成器を初期化（RAGシステムが既に構築済みの場合）
    try:
        session_id = rag_system.initialize_generator()
        print(f"RAGシステムが準備完了しました (セッション: {session_id})")
        print("質問を入力してください（'quit'で終了）:")
        
        while True:
            question = input("\nQ: ")
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if question.strip():
                result = rag_system.ask_question(question)
                print(f"A: {result['answer']}")
                
                # エラーがあった場合は表示
                if 'error' in result:
                    print(f"エラー: {result['error']}")
                    
    except Exception as e:
        print(f"システム初期化エラー: {str(e)}")
        print("先にRAGシステムを構築してください（example_usage()を実行）")


def rebuild_rag_system():
    """
    RAGシステムを再構築（新しいファイルが追加された場合など）
    """
    print("RAGシステムを再構築します...")
    
    rag_system = RAGSystem(dbutils)
    build_stats = rag_system.build_rag_system()
    
    print("再構築完了!")
    print(f"処理ファイル数: {build_stats['extraction']['total_extracted']}")
    print(f"生成チャンク数: {build_stats['chunking']['total_chunks']}")
    print(f"インデックス化ドキュメント数: {build_stats['indexing']['total_documents']}")


# ===============================================
# 実行部分
# ===============================================

if __name__ == "__main__":
    # 使用例を実行
    # 注意: 実際のDatabricks環境で実行してください
    
    print("=== RAGシステム Databricks 16.4 LTS版 ===")
    print("利用可能な機能:")
    print("- example_usage(): RAGシステムの完全な使用例")
    print("- simple_qa_interface(): 簡単な質問応答インターフェース")
    print("- rebuild_rag_system(): RAGシステムの再構築")
    
    # 実際に実行する場合は以下のコメントアウトを外してください
    # example_usage()