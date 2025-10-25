"""
RAGシステム構築クラス
ファイル処理、チャンキング、ベクトル化、インデックス構築を担当
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
import faiss
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import statistics

# ドキュメント処理ライブラリ
from docx import Document
import openpyxl
from pptx import Presentation

# 既存ライブラリ使用（絶対保護対象）
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangChainDocument

# BM25検索
from rank_bm25 import BM25Okapi

from llm_base import LLMBase


class RAGBuilder(LLMBase):
    """
    RAGシステム構築クラス
    ファイル処理からインデックス構築、成果物確認まで実行
    """
    
    # サポート対象ファイル拡張子
    SUPPORTED_EXTENSIONS = ['.docx', '.xlsx', '.pptx', '.txt']
    
    # チャンキング設定
    DEFAULT_CHUNK_SIZE = 1000
    DEFAULT_CHUNK_OVERLAP = 200
    
    def __init__(self, dbutils):
        """
        RAG構築クラスの初期化
        
        Args:
            dbutils: Databricksのdbutilsオブジェクト
        """
        super().__init__(dbutils)
        
        # 処理統計
        self.processed_files = []
        self.failed_files = []
        self.extraction_stats = {}
        self.chunking_stats = {}
        self.vectorization_stats = {}
        
        # 基本設定の実行
        self.setup_llm_client()
        self.setup_bge_m3_model()
        
        print("✅ RAGBuilder初期化完了")
    
    def build_rag_system(self) -> bool:
        """
        RAGシステムの構築メインフロー
        
        Returns:
            bool: 構築成功フラグ
        """
        try:
            print("🔄 RAGシステム構築を開始します...")
            
            # 1. ファイル処理・テキスト抽出
            print("\n【ステップ1】ファイル処理・テキスト抽出")
            if not self.process_all_files():
                print("❌ ファイル処理に失敗しました")
                return False
            
            # 2. チャンキング処理
            print("\n【ステップ2】チャンキング処理")
            if not self.chunk_texts():
                print("❌ チャンキング処理に失敗しました")
                return False
            
            # 3. ベクトル化・Faissインデックス構築
            print("\n【ステップ3】ベクトル化・Faissインデックス構築")
            if not self.create_embeddings_and_faiss_index():
                print("❌ ベクトル化・インデックス構築に失敗しました")
                return False
            
            # 4. BM25インデックス構築
            print("\n【ステップ4】BM25インデックス構築")
            if not self.build_bm25_index():
                print("❌ BM25インデックス構築に失敗しました")
                return False
            
            print("✅ RAGシステム構築が完了しました")
            return True
            
        except Exception as e:
            print(f"❌ RAGシステム構築エラー: {e}")
            return False
    
    def process_all_files(self) -> bool:
        """
        INPUT_DIRECTORYの全ファイル処理
        
        Returns:
            bool: 処理成功フラグ
        """
        try:
            print(f"📁 処理対象ディレクトリ: {self.INPUT_DIRECTORY}")
            
            if not os.path.exists(self.INPUT_DIRECTORY):
                print(f"❌ 入力ディレクトリが存在しません: {self.INPUT_DIRECTORY}")
                return False
            
            # ファイル一覧の取得（再帰的）
            target_files = []
            for root, dirs, files in os.walk(self.INPUT_DIRECTORY):
                for file in files:
                    file_path = os.path.join(root, file)
                    file_ext = os.path.splitext(file)[1].lower()
                    
                    if file_ext in self.SUPPORTED_EXTENSIONS:
                        target_files.append(file_path)
            
            print(f"📋 処理対象ファイル数: {len(target_files)}")
            
            if len(target_files) == 0:
                print("⚠️ 処理対象ファイルが見つかりません")
                return False
            
            # 各ファイルの処理
            success_count = 0
            for file_path in target_files:
                if self._process_single_file(file_path):
                    success_count += 1
            
            print(f"✅ ファイル処理完了 - 成功: {success_count}件, 失敗: {len(target_files) - success_count}件")
            
            # 処理統計の保存
            self._save_processing_stats()
            
            return success_count > 0
            
        except Exception as e:
            print(f"❌ ファイル処理エラー: {e}")
            return False
    
    def _process_single_file(self, file_path: str) -> bool:
        """
        単一ファイルの処理
        
        Args:
            file_path: ファイルパス
            
        Returns:
            bool: 処理成功フラグ
        """
        try:
            file_name = os.path.basename(file_path)
            file_ext = os.path.splitext(file_name)[1].lower()
            
            print(f"📄 処理中: {file_name}")
            
            # ファイル形式に応じた処理
            extracted_text = ""
            start_time = datetime.now()
            
            if file_ext == '.docx':
                extracted_text = self.extract_text_from_docx(file_path)
            elif file_ext == '.xlsx':
                extracted_text = self.extract_text_from_xlsx(file_path)
            elif file_ext == '.pptx':
                extracted_text = self.extract_text_from_pptx(file_path)
            elif file_ext == '.txt':
                extracted_text = self.extract_text_from_txt(file_path)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            if not extracted_text.strip():
                print(f"⚠️ テキスト抽出結果が空です: {file_name}")
                self.failed_files.append({
                    'file_path': file_path,
                    'error': 'テキスト抽出結果が空',
                    'timestamp': datetime.now().isoformat()
                })
                return False
            
            # 抽出テキストの保存
            output_file_name = os.path.splitext(file_name)[0] + '.txt'
            output_path = os.path.join(self.EXTRACTED_TEXT_DIR, output_file_name)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(extracted_text)
            
            # 処理統計の記録
            stats = {
                'file_path': file_path,
                'file_name': file_name,
                'file_extension': file_ext,
                'extracted_char_count': len(extracted_text),
                'output_path': output_path,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            
            self.processed_files.append(stats)
            self.extraction_stats[file_name] = stats
            
            print(f"  ✅ 抽出文字数: {len(extracted_text):,}文字, 処理時間: {processing_time:.2f}秒")
            return True
            
        except Exception as e:
            print(f"❌ ファイル処理エラー {file_path}: {e}")
            self.failed_files.append({
                'file_path': file_path,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            return False
    
    def extract_text_from_docx(self, file_path: str) -> str:
        """
        Wordファイルからテキスト抽出
        
        Args:
            file_path: Wordファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            doc = Document(file_path)
            text_parts = []
            
            # 段落の抽出
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    text_parts.append(paragraph.text.strip())
            
            # 表の抽出
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        if cell.text.strip():
                            row_text.append(cell.text.strip())
                    if row_text:
                        text_parts.append(" | ".join(row_text))
            
            return "\n\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"Word文書処理エラー: {e}")
    
    def extract_text_from_xlsx(self, file_path: str) -> str:
        """
        Excelファイルからテキスト抽出
        
        Args:
            file_path: Excelファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            workbook = openpyxl.load_workbook(file_path, data_only=True)
            text_parts = []
            
            for sheet_name in workbook.sheetnames:
                sheet = workbook[sheet_name]
                text_parts.append(f"=== シート: {sheet_name} ===")
                
                sheet_data = []
                for row in sheet.iter_rows(values_only=True):
                    row_data = []
                    for cell_value in row:
                        if cell_value is not None:
                            row_data.append(str(cell_value).strip())
                    
                    if row_data and any(data for data in row_data):
                        sheet_data.append(" | ".join(row_data))
                
                if sheet_data:
                    text_parts.extend(sheet_data)
                text_parts.append("")  # シート間の区切り
            
            return "\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"Excel文書処理エラー: {e}")
    
    def extract_text_from_pptx(self, file_path: str) -> str:
        """
        PowerPointファイルからテキスト抽出
        
        Args:
            file_path: PowerPointファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            presentation = Presentation(file_path)
            text_parts = []
            
            for slide_num, slide in enumerate(presentation.slides, 1):
                text_parts.append(f"=== スライド {slide_num} ===")
                
                # スライド内のテキスト抽出
                slide_texts = []
                for shape in slide.shapes:
                    if hasattr(shape, "text") and shape.text.strip():
                        slide_texts.append(shape.text.strip())
                
                if slide_texts:
                    text_parts.extend(slide_texts)
                
                # ノートの抽出
                if slide.notes_slide and slide.notes_slide.notes_text_frame:
                    notes_text = slide.notes_slide.notes_text_frame.text.strip()
                    if notes_text:
                        text_parts.append(f"[ノート] {notes_text}")
                
                text_parts.append("")  # スライド間の区切り
            
            return "\n".join(text_parts)
            
        except Exception as e:
            raise Exception(f"PowerPoint文書処理エラー: {e}")
    
    def extract_text_from_txt(self, file_path: str) -> str:
        """
        テキストファイルからテキスト抽出
        
        Args:
            file_path: テキストファイルパス
            
        Returns:
            str: 抽出されたテキスト
        """
        try:
            # エンコーディングの自動判別
            encodings = ['utf-8', 'shift_jis', 'cp932', 'euc-jp', 'iso-2022-jp']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        return f.read()
                except UnicodeDecodeError:
                    continue
            
            # 全て失敗した場合はエラー
            raise Exception("サポートされていないエンコーディングです")
            
        except Exception as e:
            raise Exception(f"テキストファイル処理エラー: {e}")
    
    def chunk_texts(self) -> bool:
        """
        抽出テキストのチャンキング処理
        
        Returns:
            bool: 処理成功フラグ
        """
        try:
            print(f"📝 チャンキング処理開始")
            
            # TextSplitterの設定（日本語対応）
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.DEFAULT_CHUNK_SIZE,
                chunk_overlap=self.DEFAULT_CHUNK_OVERLAP,
                separators=['\n\n', '\n', '。', '、', ' ', ''],
                length_function=len
            )
            
            # 抽出テキストファイルの取得
            extracted_files = [f for f in os.listdir(self.EXTRACTED_TEXT_DIR) 
                             if f.endswith('.txt')]
            
            if not extracted_files:
                print("❌ 抽出テキストファイルが見つかりません")
                return False
            
            all_chunks = []
            chunk_metadata = []
            
            for text_file in extracted_files:
                file_path = os.path.join(self.EXTRACTED_TEXT_DIR, text_file)
                
                # テキスト読み込み
                with open(file_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
                
                if not text_content.strip():
                    continue
                
                # チャンキング実行
                chunks = text_splitter.split_text(text_content)
                
                # チャンクメタデータの作成
                source_file = os.path.splitext(text_file)[0]
                for i, chunk in enumerate(chunks):
                    if chunk.strip():
                        chunk_id = f"{source_file}_chunk_{i:04d}"
                        
                        chunk_info = {
                            'chunk_id': chunk_id,
                            'source_file': source_file,
                            'chunk_index': i,
                            'content': chunk.strip(),
                            'char_count': len(chunk.strip()),
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        all_chunks.append(chunk.strip())
                        chunk_metadata.append(chunk_info)
                
                print(f"  ✅ {text_file}: {len(chunks)}チャンク生成")
            
            if not all_chunks:
                print("❌ 有効なチャンクが生成されませんでした")
                return False
            
            # チャンク統計の計算
            chunk_sizes = [len(chunk) for chunk in all_chunks]
            self.chunking_stats = {
                'total_chunks': len(all_chunks),
                'chunk_size_stats': {
                    'min': min(chunk_sizes),
                    'max': max(chunk_sizes),
                    'mean': statistics.mean(chunk_sizes),
                    'median': statistics.median(chunk_sizes)
                },
                'chunk_overlap': self.DEFAULT_CHUNK_OVERLAP,
                'timestamp': datetime.now().isoformat()
            }
            
            # チャンクデータの保存
            chunks_file = os.path.join(self.CHUNKED_TEXT_DIR, 'chunks_data.json')
            metadata_file = os.path.join(self.CHUNKED_TEXT_DIR, 'chunk_metadata.json')
            
            chunk_data = {
                'chunks': all_chunks,
                'metadata': chunk_metadata,
                'stats': self.chunking_stats
            }
            
            self.save_json(chunk_data, chunks_file)
            
            print(f"✅ チャンキング完了 - 総チャンク数: {len(all_chunks)}")
            print(f"   平均チャンクサイズ: {self.chunking_stats['chunk_size_stats']['mean']:.0f}文字")
            
            return True
            
        except Exception as e:
            print(f"❌ チャンキング処理エラー: {e}")
            return False
    
    def create_embeddings_and_faiss_index(self) -> bool:
        """
        ベクトル化とFaissインデックス構築
        
        Returns:
            bool: 処理成功フラグ
        """
        try:
            print("🔢 ベクトル化・Faissインデックス構築開始")
            
            # チャンクデータの読み込み
            chunks_file = os.path.join(self.CHUNKED_TEXT_DIR, 'chunks_data.json')
            chunk_data = self.load_json(chunks_file)
            
            if not chunk_data:
                print("❌ チャンクデータが見つかりません")
                return False
            
            chunks = chunk_data['chunks']
            metadata = chunk_data['metadata']
            
            print(f"📊 ベクトル化対象チャンク数: {len(chunks)}")
            
            # バッチ処理でベクトル化
            batch_size = 10  # BGE-M3の処理能力に応じて調整
            all_embeddings = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                print(f"  処理中: {i + 1}-{min(i + batch_size, len(chunks))} / {len(chunks)}")
                
                # pandas DataFrameでの予測実行
                batch_df = pd.DataFrame({"input": batch_chunks})
                response = self.bge_m3_model.predict(batch_df)
                
                # ベクトル抽出
                batch_embeddings = self.extract_embedding_vectors(response)
                
                if len(batch_embeddings) != len(batch_chunks):
                    print(f"❌ ベクトル数とチャンク数が一致しません: {len(batch_embeddings)} vs {len(batch_chunks)}")
                    return False
                
                all_embeddings.extend(batch_embeddings)
            
            print(f"✅ ベクトル化完了: {len(all_embeddings)}個のベクトル")
            
            # Faissインデックスの構築
            embedding_matrix = np.array(all_embeddings, dtype=np.float32)
            
            # Inner Product（内積）インデックスを使用
            index = faiss.IndexFlatIP(embedding_matrix.shape[1])
            
            # ベクトルの正規化（コサイン類似度のため）
            faiss.normalize_L2(embedding_matrix)
            
            # インデックスに追加
            index.add(embedding_matrix)
            
            print(f"✅ Faissインデックス構築完了: {index.ntotal}個のベクトル, {index.d}次元")
            
            # インデックスとメタデータの保存
            index_file = os.path.join(self.FAISS_INDEX_DIR, 'faiss.index')
            documents_file = os.path.join(self.FAISS_INDEX_DIR, 'documents.json')
            config_file = os.path.join(self.FAISS_INDEX_DIR, 'index_config.json')
            
            # Faissインデックス保存
            faiss.write_index(index, index_file)
            
            # ドキュメントメタデータ保存
            documents_data = {
                'chunks': chunks,
                'metadata': metadata,
                'total_count': len(chunks),
                'vector_dimension': embedding_matrix.shape[1],
                'timestamp': datetime.now().isoformat()
            }
            self.save_json(documents_data, documents_file)
            
            # 設定情報保存
            config_data = {
                'index_type': 'IndexFlatIP',
                'vector_dimension': embedding_matrix.shape[1],
                'total_vectors': len(all_embeddings),
                'chunk_size': self.DEFAULT_CHUNK_SIZE,
                'chunk_overlap': self.DEFAULT_CHUNK_OVERLAP,
                'model_name': 'bge_m3',
                'timestamp': datetime.now().isoformat()
            }
            self.save_json(config_data, config_file)
            
            # ベクトル化統計
            self.vectorization_stats = {
                'total_vectors': len(all_embeddings),
                'vector_dimension': embedding_matrix.shape[1],
                'index_file_size_mb': self.get_file_size_mb(index_file),
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"   インデックスファイルサイズ: {self.vectorization_stats['index_file_size_mb']:.2f}MB")
            
            return True
            
        except Exception as e:
            print(f"❌ ベクトル化・インデックス構築エラー: {e}")
            return False
    
    def build_bm25_index(self) -> bool:
        """
        BM25インデックスの構築
        
        Returns:
            bool: 処理成功フラグ
        """
        try:
            print("📚 BM25インデックス構築開始")
            
            # ドキュメントデータの読み込み
            documents_file = os.path.join(self.FAISS_INDEX_DIR, 'documents.json')
            documents_data = self.load_json(documents_file)
            
            if not documents_data:
                print("❌ ドキュメントデータが見つかりません")
                return False
            
            chunks = documents_data['chunks']
            
            # 簡易的な日本語トークナイゼーション（文字レベル）
            # 実際の運用では、MeCab等の形態素解析器を使用することを推奨
            tokenized_chunks = []
            for chunk in chunks:
                # 文字レベルでの分割（ひらがな、カタカナ、漢字、英数字を保持）
                tokens = []
                current_token = ""
                
                for char in chunk:
                    if char.isalnum() or char in 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんがぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽゃゅょっアイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲンガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポャュョッ':
                        current_token += char
                    else:
                        if current_token:
                            tokens.append(current_token)
                            current_token = ""
                        if char.strip():  # 空白文字以外の記号など
                            tokens.append(char)
                
                if current_token:
                    tokens.append(current_token)
                
                # 2文字以上のトークンのみを使用
                filtered_tokens = [token for token in tokens if len(token) >= 2]
                tokenized_chunks.append(filtered_tokens)
            
            # BM25インデックスの構築
            bm25 = BM25Okapi(tokenized_chunks)
            
            # BM25インデックスの保存
            bm25_file = os.path.join(self.FAISS_INDEX_DIR, 'bm25_index.pkl')
            with open(bm25_file, 'wb') as f:
                pickle.dump(bm25, f)
            
            # 語彙統計の計算
            all_tokens = set()
            for tokens in tokenized_chunks:
                all_tokens.update(tokens)
            
            vocab_size = len(all_tokens)
            
            print(f"✅ BM25インデックス構築完了")
            print(f"   対象文書数: {len(tokenized_chunks)}")
            print(f"   語彙数: {vocab_size}")
            print(f"   インデックスファイルサイズ: {self.get_file_size_mb(bm25_file):.2f}MB")
            
            # BM25設定情報の保存
            bm25_config = {
                'document_count': len(tokenized_chunks),
                'vocabulary_size': vocab_size,
                'tokenization_method': 'character_level',
                'min_token_length': 2,
                'timestamp': datetime.now().isoformat()
            }
            
            bm25_config_file = os.path.join(self.FAISS_INDEX_DIR, 'bm25_config.json')
            self.save_json(bm25_config, bm25_config_file)
            
            return True
            
        except Exception as e:
            print(f"❌ BM25インデックス構築エラー: {e}")
            return False
    
    def confirm_build_results(self) -> Dict[str, Any]:
        """
        RAG構築成果物の包括的確認・検証機能
        
        Returns:
            Dict[str, Any]: 確認結果の詳細情報
        """
        try:
            print("🔍 成果物確認を開始します...")
            
            confirmation_results = {
                'timestamp': datetime.now().isoformat(),
                'processed_files': self._check_processed_files(),
                'extracted_texts': self._check_extracted_texts(),
                'chunking_results': self._check_chunking_results(),
                'vector_indexes': self._check_vector_indexes(),
                'directory_structure': self._check_directory_structure(),
                'quality_metrics': self._check_quality_metrics(),
                'errors_warnings': self._collect_errors_warnings()
            }
            
            # 確認結果の表示
            self.display_confirmation_results(confirmation_results)
            
            # 確認結果の保存
            self._save_confirmation_results(confirmation_results)
            
            return confirmation_results
            
        except Exception as e:
            print(f"❌ 成果物確認エラー: {e}")
            return {}
    
    def _check_processed_files(self) -> Dict[str, Any]:
        """処理済みファイル情報の確認"""
        try:
            # ファイル形式別の統計
            extension_stats = {}
            for file_info in self.processed_files:
                ext = file_info['file_extension']
                if ext not in extension_stats:
                    extension_stats[ext] = {'count': 0, 'total_chars': 0}
                extension_stats[ext]['count'] += 1
                extension_stats[ext]['total_chars'] += file_info['extracted_char_count']
            
            return {
                'total_processed': len(self.processed_files),
                'total_failed': len(self.failed_files),
                'extension_breakdown': extension_stats,
                'processing_times': [f['processing_time'] for f in self.processed_files],
                'failed_files_details': self.failed_files
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_extracted_texts(self) -> Dict[str, Any]:
        """抽出テキスト情報の確認"""
        try:
            if not os.path.exists(self.EXTRACTED_TEXT_DIR):
                return {'error': '抽出テキストディレクトリが存在しません'}
            
            text_files = [f for f in os.listdir(self.EXTRACTED_TEXT_DIR) if f.endswith('.txt')]
            total_chars = 0
            
            for text_file in text_files:
                file_path = os.path.join(self.EXTRACTED_TEXT_DIR, text_file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        total_chars += len(content)
                except:
                    pass
            
            return {
                'extracted_file_count': len(text_files),
                'total_extracted_chars': total_chars,
                'directory_size_mb': sum(self.get_file_size_mb(os.path.join(self.EXTRACTED_TEXT_DIR, f)) 
                                       for f in text_files)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_chunking_results(self) -> Dict[str, Any]:
        """チャンキング結果の確認"""
        try:
            chunks_file = os.path.join(self.CHUNKED_TEXT_DIR, 'chunks_data.json')
            chunk_data = self.load_json(chunks_file)
            
            if not chunk_data:
                return {'error': 'チャンクデータが見つかりません'}
            
            return {
                'total_chunks': len(chunk_data['chunks']),
                'chunk_size_stats': chunk_data['stats']['chunk_size_stats'],
                'chunk_overlap': chunk_data['stats']['chunk_overlap']
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _check_vector_indexes(self) -> Dict[str, Any]:
        """ベクトルインデックス情報の確認"""
        try:
            faiss_file = os.path.join(self.FAISS_INDEX_DIR, 'faiss.index')
            bm25_file = os.path.join(self.FAISS_INDEX_DIR, 'bm25_index.pkl')
            documents_file = os.path.join(self.FAISS_INDEX_DIR, 'documents.json')
            
            results = {
                'faiss_index_exists': os.path.exists(faiss_file),
                'bm25_index_exists': os.path.exists(bm25_file),
                'documents_file_exists': os.path.exists(documents_file)
            }
            
            if results['faiss_index_exists']:
                index = faiss.read_index(faiss_file)
                results['faiss_vector_count'] = index.ntotal
                results['faiss_vector_dimension'] = index.d
                results['faiss_file_size_mb'] = self.get_file_size_mb(faiss_file)
            
            if results['bm25_index_exists']:
                results['bm25_file_size_mb'] = self.get_file_size_mb(bm25_file)
            
            if results['documents_file_exists']:
                documents_data = self.load_json(documents_file)
                if documents_data:
                    results['document_count'] = documents_data['total_count']
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def _check_directory_structure(self) -> Dict[str, Any]:
        """ディレクトリ構造の確認"""
        try:
            directories = [
                self.INPUT_DIRECTORY,
                self.EXTRACTED_TEXT_DIR,
                self.CHUNKED_TEXT_DIR,
                self.FAISS_INDEX_DIR,
                self.CONVERSATION_HISTORY_DIR,
                self.BUILD_LOGS_DIR
            ]
            
            dir_info = {}
            for directory in directories:
                dir_name = os.path.basename(directory)
                dir_info[dir_name] = {
                    'exists': os.path.exists(directory),
                    'file_count': len(os.listdir(directory)) if os.path.exists(directory) else 0
                }
            
            return dir_info
        except Exception as e:
            return {'error': str(e)}
    
    def _check_quality_metrics(self) -> Dict[str, Any]:
        """品質メトリクスの確認"""
        try:
            chunks_file = os.path.join(self.CHUNKED_TEXT_DIR, 'chunks_data.json')
            chunk_data = self.load_json(chunks_file)
            
            if not chunk_data:
                return {'error': 'チャンクデータが見つかりません'}
            
            chunks = chunk_data['chunks']
            
            # 品質チェック
            empty_chunks = sum(1 for chunk in chunks if not chunk.strip())
            too_short_chunks = sum(1 for chunk in chunks if len(chunk.strip()) < 50)
            too_long_chunks = sum(1 for chunk in chunks if len(chunk.strip()) > 2000)
            
            return {
                'empty_chunks': empty_chunks,
                'too_short_chunks': too_short_chunks,
                'too_long_chunks': too_long_chunks,
                'quality_score': max(0, 100 - (empty_chunks + too_short_chunks) / len(chunks) * 100)
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _collect_errors_warnings(self) -> Dict[str, List[str]]:
        """エラー・警告の収集"""
        errors = []
        warnings = []
        
        # 失敗ファイルがある場合
        if self.failed_files:
            errors.append(f"{len(self.failed_files)}個のファイル処理が失敗しました")
        
        # チャンク数とベクトル数の整合性確認
        try:
            chunks_file = os.path.join(self.CHUNKED_TEXT_DIR, 'chunks_data.json')
            documents_file = os.path.join(self.FAISS_INDEX_DIR, 'documents.json')
            
            chunk_data = self.load_json(chunks_file)
            documents_data = self.load_json(documents_file)
            
            if chunk_data and documents_data:
                chunk_count = len(chunk_data['chunks'])
                doc_count = documents_data['total_count']
                
                if chunk_count != doc_count:
                    errors.append(f"チャンク数とドキュメント数が不一致: {chunk_count} vs {doc_count}")
        except:
            warnings.append("チャンク・ドキュメント整合性の確認に失敗しました")
        
        return {'errors': errors, 'warnings': warnings}
    
    def display_confirmation_results(self, results: Dict[str, Any]) -> None:
        """確認結果の表示"""
        print("\n" + "="*60)
        print("📋 RAG構築成果物確認結果")
        print("="*60)
        
        # 1. ファイル処理結果
        if 'processed_files' in results:
            pf = results['processed_files']
            print(f"【ファイル処理結果】")
            print(f"処理成功: {pf.get('total_processed', 0)}ファイル")
            print(f"処理失敗: {pf.get('total_failed', 0)}ファイル")
            
            if 'extension_breakdown' in pf:
                print("ファイル形式別:")
                for ext, stats in pf['extension_breakdown'].items():
                    print(f"  {ext}: {stats['count']}ファイル, {stats['total_chars']:,}文字")
        
        # 2. チャンキング結果
        if 'chunking_results' in results:
            cr = results['chunking_results']
            print(f"\n【チャンキング結果】")
            print(f"総チャンク数: {cr.get('total_chunks', 0)}個")
            
            if 'chunk_size_stats' in cr:
                stats = cr['chunk_size_stats']
                print(f"チャンクサイズ - 最小: {stats.get('min', 0)}文字, 最大: {stats.get('max', 0)}文字")
                print(f"              平均: {stats.get('mean', 0):.0f}文字, 中央値: {stats.get('median', 0)}文字")
        
        # 3. インデックス情報
        if 'vector_indexes' in results:
            vi = results['vector_indexes']
            print(f"\n【インデックス情報】")
            print(f"Faissベクトル数: {vi.get('faiss_vector_count', 0)}個")
            print(f"ベクトル次元数: {vi.get('faiss_vector_dimension', 0)}次元")
            print(f"Faissファイルサイズ: {vi.get('faiss_file_size_mb', 0):.2f}MB")
            print(f"BM25ファイルサイズ: {vi.get('bm25_file_size_mb', 0):.2f}MB")
        
        # 4. エラー・警告
        if 'errors_warnings' in results:
            ew = results['errors_warnings']
            errors = ew.get('errors', [])
            warnings = ew.get('warnings', [])
            
            if errors or warnings:
                print(f"\n【エラー・警告】")
                for error in errors:
                    print(f"エラー: {error}")
                for warning in warnings:
                    print(f"警告: {warning}")
            else:
                print(f"\n✅ エラー・警告なし")
        
        # 5. 準備状況
        faiss_ready = results.get('vector_indexes', {}).get('faiss_index_exists', False)
        bm25_ready = results.get('vector_indexes', {}).get('bm25_index_exists', False)
        
        print(f"\n【回答生成準備状況】")
        print(f"ベクトル検索: {'✅ 準備完了' if faiss_ready else '❌ 未完了'}")
        print(f"テキスト検索: {'✅ 準備完了' if bm25_ready else '❌ 未完了'}")
        print(f"ハイブリッド検索: {'✅ 準備完了' if (faiss_ready and bm25_ready) else '❌ 未完了'}")
        
        print("="*60)
    
    def _save_confirmation_results(self, results: Dict[str, Any]) -> None:
        """確認結果の保存"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 確認結果の保存
            confirmation_file = os.path.join(self.BUILD_LOGS_DIR, f'confirmation_{timestamp}.json')
            self.save_json(results, confirmation_file)
            
            # 構築サマリーの保存
            summary = {
                'build_timestamp': results['timestamp'],
                'total_processed_files': results.get('processed_files', {}).get('total_processed', 0),
                'total_chunks': results.get('chunking_results', {}).get('total_chunks', 0),
                'faiss_vectors': results.get('vector_indexes', {}).get('faiss_vector_count', 0),
                'ready_for_retrieval': (
                    results.get('vector_indexes', {}).get('faiss_index_exists', False) and
                    results.get('vector_indexes', {}).get('bm25_index_exists', False)
                )
            }
            
            summary_file = os.path.join(self.BUILD_LOGS_DIR, f'build_summary_{timestamp}.json')
            self.save_json(summary, summary_file)
            
        except Exception as e:
            print(f"❌ 確認結果保存エラー: {e}")
    
    def _save_processing_stats(self) -> None:
        """処理統計の保存"""
        try:
            stats = {
                'processed_files': self.processed_files,
                'failed_files': self.failed_files,
                'extraction_stats': self.extraction_stats,
                'chunking_stats': self.chunking_stats,
                'vectorization_stats': self.vectorization_stats,
                'timestamp': datetime.now().isoformat()
            }
            
            stats_file = os.path.join(self.BUILD_LOGS_DIR, 'processing_stats.json')
            self.save_json(stats, stats_file)
            
        except Exception as e:
            print(f"❌ 処理統計保存エラー: {e}")