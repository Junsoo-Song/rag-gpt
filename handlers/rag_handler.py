"""
RAG 핸들러
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
        self.setup_llm()
    
    def setup_llm(self):
        """LLM 설정"""
        self.llm = ChatGroq(
            model=self.config.get("model"),
            temperature=self.config.get("temperature", 0.3)
        )
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "당신은 문서 기반 AI 어시스턴트입니다. 한국어로 답변하세요."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("user", "질문: {question}\n\n참고 문서:\n{context}")
        ])
    
    def process_pdf(self, pdf_path: Path):
        """PDF 처리 및 벡터화"""
        # PDF 로드
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        
        # 청크 분할
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.config.get("chunk_size", 500),
            chunk_overlap=self.config.get("chunk_overlap", 50)
        )
        chunks = splitter.split_documents(docs)
        
        # 임베딩
        embedding = HuggingFaceEmbeddings(
            model_name="intfloat/multilingual-e5-small"
        )
        
        # 벡터스토어 생성
        self.vectorstore = FAISS.from_documents(chunks, embedding)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 3)}
        )
        
        return self.vectorstore
    
    def query(self, question: str, chat_history: List = None) -> str:
        """질문 처리"""
        if not self.retriever:
            return "⚠️ PDF를 먼저 로드해주세요."
        
        chat_history = chat_history or []
        
        # 문서 검색
        docs = self.retriever.invoke(question)
        context = "\n".join(d.page_content for d in docs)
        
        # 체인 실행
        chain = (
            {
                "context": lambda x: context,
                "question": itemgetter("question"),
                "chat_history": itemgetter("chat_history")
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
        """저장된 벡터스토어 로드"""
        self.vectorstore = vectorstore
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.get("top_k", 3)}
        )

