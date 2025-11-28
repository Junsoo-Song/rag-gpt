# RAG-GPT: 문서 기반 AI 어시스턴트

Shell-GPT 스타일의 RAG(Retrieval Augmented Generation) 기반 문서 질의응답 시스템

## 🚀 Features

- PDF 문서 업로드 및 벡터화
- 대화형 REPL 모드
- 웹 인터페이스 지원 (Gradio)
- 세션 관리 및 히스토리
- 다중 LLM 모델 지원

## 📦 Installation

```bash
# 1. 저장소 클론
git clone https://github.com/your-username/rag-gpt.git
cd rag-gpt

# 2. 가상환경 생성 (권장)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. 패키지 설치
pip install -r requirements.txt

# 4. 환경변수 설정
cp .env.example .env
# .env 파일을 열어서 GROQ_API_KEY 입력

