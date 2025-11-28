"""
RAG 핸들러 - 메타데이터 및 프롬프트 개선
"""
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

class RAGHandler:
    """RAG 처리 핸들러"""
    
    def __init__(self, config):
        self.config = config
        self.vectorstore = None
        self.retriever = None
        self.loaded_pdfs = []
        self.embedding = None
        self.setup_llm()
        self.setup_embedding()
    
    def setup_embedding(self):
        self.embedding = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small"
        )
    
    def setup_llm(self):
        self.llm = ChatGroq(
            model=self.config.get("model"),
            temperature=self.config.get("temperature", 0.3)
        )
        
        # 프롬프트 템플릿 수정 - 파일 목록 정보 추가
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 문서 기반 AI 어시스턴트입니다. 
            현재 로드된 문서 목록: {file_list}
            
            답변 작성 규칙:
            1. 반드시 한국어로 답변하세요.
            2. 질문과 관련된 문서의 내용을 바탕으로 답변하세요.
            3. 문서에 없는 내용은 모른다고 하세요.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "질문: {question}\n\n참고 문서:\n{context}")
        ])
    
    def process_pdf(self, pdf_path: Path) -> int:
        """단일 PDF 처리"""
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50)
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
            "total_chunks": 0
        }
        
        for pdf_path in pdf_paths:
            try:
                chunks_count = self.process_pdf(pdf_path)
                results["success"].append({
                    "file": pdf_path.name,
                    "chunks": chunks_count
                })
                results["total_chunks"] += chunks_count
            except Exception as e:
                results["failed"].append({
                    "file": pdf_path.name,
                    "error": str(e)
                })
        
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
        
        # 문서 검색
        docs = self.retriever.invoke(question)
        
        # 출처 포함 컨텍스트
        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            context_parts.append(f"[출처: {source}]\n{doc.page_content}")
        
        context = "\n\n".join(context_parts)
        
        # 체인 실행 (file_list 변수 추가)
        chain = (
            {
                "context": lambda x: context,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history"),
                "file_list": lambda x: file_info  # 여기서 파일 목록 전달
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        
        response = chain.invoke({
            "question": question,
            "chat_history": chat_history
        })
        
        return response
    
    def load_vectorstore(self, vectorstore):
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 3)}
        )
