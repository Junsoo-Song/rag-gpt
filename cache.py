"""
벡터스토어 캐시 관리
"""
from pathlib import Path
import pickle
import hashlib

class VectorCache:
    """벡터스토어 캐시"""
    
    def __init__(self):
        self.cache_dir = Path.home() / ".rag_gpt" / "vectors"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self, pdf_path: Path) -> Path:
        """캐시 파일 경로 생성"""
        # PDF 파일의 해시값으로 고유 이름 생성
        hash_obj = hashlib.md5(str(pdf_path).encode())
        cache_name = f"{pdf_path.stem}_{hash_obj.hexdigest()[:8]}.pkl"
        return self.cache_dir / cache_name
    
    def exists(self, pdf_path: Path) -> bool:
        """캐시 존재 여부 확인"""
        return self._get_cache_path(pdf_path).exists()
    
    def save(self, pdf_path: Path, vectorstore):
        """벡터스토어 저장"""
        cache_path = self._get_cache_path(pdf_path)
        vectorstore.save_local(str(cache_path))
    
    def load(self, pdf_path: Path):
        """벡터스토어 로드"""
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        
        cache_path = self._get_cache_path(pdf_path)
        embedding = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small"
        )
        return FAISS.load_local(
            str(cache_path), 
            embedding,
            allow_dangerous_deserialization=True
        )

