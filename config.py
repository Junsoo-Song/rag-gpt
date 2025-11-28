"""
설정 관리 모듈
"""
import json
from pathlib import Path
from typing import Any

class Config:
    """설정 관리 클래스"""
    
    def __init__(self):
        self.config_dir = Path.home() / ".rag_gpt"
        self.config_file = self.config_dir / "config.json"
        self.ensure_dirs()
        self.load()
    
    def ensure_dirs(self):
        """필요한 디렉토리 생성"""
        self.config_dir.mkdir(exist_ok=True)
        (self.config_dir / "sessions").mkdir(exist_ok=True)
        (self.config_dir / "vectors").mkdir(exist_ok=True)
    
    def load(self):
        """설정 로드"""
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                self.data = json.load(f)
        else:
            self.data = {
                "api_key": "",
                "model": "llama-3.3-70b-versatile",
                "temperature": 0.3,
                "chunk_size": 500,
                "chunk_overlap": 50,
                "top_k": 3
            }
            self.save()
    
    def save(self):
        """설정 저장"""
        with open(self.config_file, 'w') as f:
            json.dump(self.data, f, indent=2)
    
    def get(self, key: str, default: Any = None) -> Any:
        """설정 값 가져오기"""
        return self.data.get(key, default)
    
    def set(self, key: str, value: Any):
        """설정 값 설정"""
        self.data[key] = value
        self.save()

