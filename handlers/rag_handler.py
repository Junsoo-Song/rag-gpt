"""
RAG 핸들러 - 메타데이터, 프롬프트 및 언어 자동 선택
"""
from pathlib import Path
from typing import List
from operator import itemgetter

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


class RAGHandler:
    """RAG 처리 핸들러"""

    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.loaded_pdfs: List[str] = []
        self.embedding = None
        self.setup_llm()
        self.setup_embedding()

    def setup_embedding(self):
        """임베딩 모델 설정"""
        self.embedding = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small"
        )

    def setup_llm(self):
        """LLM 설정"""
        self.llm = ChatGroq(
            model=self.config.get("model"),
            temperature=self.config.get("temperature", 0.3),
        )

        # 언어를 코드에서 결정해서 answer_language로 넘김
        # answer_language 값은 "Korean" 또는 "English"로 넘기겠습니다.
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a document-grounded AI assistant.

The user wants the answer in: {answer_language}

Language rules:
- You MUST respond entirely in {answer_language}.
- Do NOT use any other language.
- If technical terms appear in another language in the documents (e.g., gene names, NGS, etc.), you may keep those terms as-is, but the surrounding explanation must be in {answer_language} only.

Content rules:
- Base your answers only on the provided documents and chat history.
- If the documents and history do not contain enough information, honestly say that you don't know, instead of hallucinating.
""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                (
                    "user",
                    "Question: {question}\n\nReference documents:\n{context}",
                ),
            ]
        )

    # ---- 언어 감지 함수 추가 ----
    def _detect_language(self, text: str) -> str:
        """
        매우 단순한 언어 감지:
        - 한글 문자가 더 많으면 'Korean'
        - 알파벳이 하나라도 있으면 'English'
        - 그 외에는 기본 'Korean'
        """
        korean_count = sum("\uac00" <= ch <= "\ud7a3" for ch in text)
        latin_count = sum(
            ("a" <= ch <= "z") or ("A" <= ch <= "Z") for ch in text
        )

        if korean_count > latin_count:
            return "Korean"
        if latin_count > 0:
            return "English"
        # 기본값: 한국어
        return "Korean"

    def process_pdf(self, pdf_path: Path) -> int:
        """단일 PDF 처리"""
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50),
        )
        chunks = splitter.split_documents(docs)

        # 메타데이터 추가
        for chunk in chunks:
            chunk.metadata["source_file"] = pdf_path.name

        if self.vectorstore is None:
            self.vectorstore = FAISS.from_documents(chunks, self.embedding)
        else:
            new_vectorstore = FAISS.from_documents(chunks, self.embedding)
            self.vectorstore.merge_from(new_vectorstore)

        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 3)}
        )

        if pdf_path.name not in self.loaded_pdfs:
            self.loaded_pdfs.append(pdf_path.name)

        return len(chunks)

    def process_multiple_pdfs(self, pdf_paths: List[Path]) -> dict:
        """다중 PDF 처리"""
        results = {
            "success": [],
            "failed": [],
            "total_chunks": 0,
        }

        for pdf_path in pdf_paths:
            try:
                chunks_count = self.process_pdf(pdf_path)
                results["success"].append(
                    {
                        "file": pdf_path.name,
                        "chunks": chunks_count,
                    }
                )
                results["total_chunks"] += chunks_count
            except Exception as e:
                results["failed"].append(
                    {
                        "file": pdf_path.name,
                        "error": str(e),
                    }
                )

        return results

    def clear_vectorstore(self):
        self.vectorstore = None
        self.retriever = None
        self.loaded_pdfs = []

    def get_loaded_pdfs(self) -> List[str]:
        return self.loaded_pdfs

    def query(self, question: str, chat_history: List = None) -> str:
        """질문 처리"""
        if not self.retriever:
            return "⚠️ PDF를 먼저 로드해주세요."

        chat_history = chat_history or []

        # 로드된 파일 목록 문자열 생성
        file_list_str = ", ".join(self.loaded_pdfs) if self.loaded_pdfs else "없음"
        file_count = len(self.loaded_pdfs)
        file_info = f"{file_count}개 파일 ({file_list_str})"

        # 질문 언어 감지
        answer_language = self._detect_language(question)

        # 문서 검색
        docs = self.retriever.invoke(question)

        # 출처 포함 컨텍스트
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            context_parts.append(f"[출처: {source}]\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        # 체인 실행 (file_list, answer_language 변수 추가)
        chain = (
            {
                "context": lambda x: context,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "file_list": lambda x: file_info,
                "answer_language": lambda x: answer_language,
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        response = chain.invoke(
            {
                "question": question,
                "chat_history": chat_history,
            }
        )

        return response

    def load_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 3)}
        )
