"""
웹 인터페이스 모듈 (Gradio) - 다중 PDF 지원
"""
import gradio as gr
from pathlib import Path
from typing import List, Optional
import json

class WebInterface:
    """Gradio 웹 인터페이스 - 다중 PDF 지원"""
    
    def __init__(self, rag_gpt_instance):
        self.rag = rag_gpt_instance
        
    def create_interface(self):
        """Gradio 인터페이스 생성"""
        
        with gr.Blocks() as demo:
            gr.Markdown("# 🤖 RAG-GPT: 문서 기반 AI 어시스턴트")
            
            with gr.Tab("💬 대화"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 다중 파일 업로드 지원
                        pdf_files = gr.File(
                            label="📄 PDF 업로드 (여러 개 선택 가능)",
                            file_types=[".pdf"],
                            file_count="multiple"
                        )
                        upload_btn = gr.Button("📥 문서 로드", variant="primary")
                        clear_docs_btn = gr.Button("🗑️ 문서 초기화", variant="secondary")
                        
                        status = gr.Textbox(
                            label="상태",
                            value="문서를 업로드해주세요.",
                            interactive=False,
                            lines=3
                        )
                        
                        # 로드된 PDF 목록
                        loaded_pdfs = gr.Textbox(
                            label="📚 로드된 문서",
                            value="없음",
                            interactive=False,
                            lines=5
                        )
                        
                        gr.Markdown("### ⚙️ 설정")
                        model_dropdown = gr.Dropdown(
                            choices=[
                                "llama-3.3-70b-versatile",
                                "mixtral-8x7b-32768",
                                "llama3-8b-8192"
                            ],
                            value=self.rag.config.get("model"),
                            label="모델 선택"
                        )
                        temperature = gr.Slider(
                            0, 1, 
                            value=self.rag.config.get("temperature", 0.3),
                            label="Temperature"
                        )
                        
                    with gr.Column(scale=2):
                        chatbot = gr.Chatbot(
                            label="대화",
                            height=500
                        )
                        msg = gr.Textbox(
                            label="질문 입력",
                            placeholder="질문을 입력하세요...",
                            lines=2
                        )
                        with gr.Row():
                            submit = gr.Button("🚀 전송", variant="primary")
                            clear = gr.Button("🗑️ 대화 초기화")
            
            with gr.Tab("📚 세션 관리"):
                with gr.Row():
                    session_name = gr.Textbox(
                        label="세션 이름",
                        placeholder="session1"
                    )
                    save_btn = gr.Button("💾 세션 저장")
                    load_btn = gr.Button("📂 세션 로드")
                
                sessions_display = gr.Textbox(
                    label="저장된 세션",
                    lines=10,
                    interactive=False
                )
                refresh_btn = gr.Button("🔄 새로고침")
            
            with gr.Tab("ℹ️ 정보"):
                gr.Markdown("""
                ## 사용법
                
                1. **PDF 업로드**: 좌측 패널에서 PDF 파일을 선택하고 '문서 로드' 클릭
                   - **여러 PDF 동시 선택 가능** (Ctrl+클릭 또는 Shift+클릭)
                2. **질문하기**: 채팅창에 질문을 입력하고 전송
                3. **세션 저장**: 대화 내용을 저장하려면 '세션 관리' 탭에서 저장
                
                ## 기능
                - 다중 PDF 동시 로드
                - 문서 출처 표시
                - 대화 세션 관리
                """)
            
            # 이벤트 핸들러
            def process_pdfs(files):
                """여러 PDF 처리"""
                if not files:
                    return "❌ 파일을 선택해주세요.", "없음"
                
                try:
                    # 파일 경로 추출
                    pdf_paths = []
                    for file in files:
                        file_path = file.name if hasattr(file, 'name') else file
                        pdf_paths.append(Path(file_path))
                    
                    # 다중 PDF 로드
                    results = self.rag.rag_handler.process_multiple_pdfs(pdf_paths)
                    
                    # 상태 메시지 생성
                    status_msg = ""
                    for success in results["success"]:
                        status_msg += f"✅ {success['file']}: {success['chunks']}개 청크\n"
                    
                    for failed in results["failed"]:
                        status_msg += f"❌ {failed['file']}: {failed['error']}\n"
                    
                    status_msg += f"\n총 {results['total_chunks']}개 청크 로드됨"
                    
                    # 로드된 PDF 목록
                    loaded_list = "\n".join([f"📄 {pdf}" for pdf in self.rag.get_loaded_pdfs()])
                    if not loaded_list:
                        loaded_list = "없음"
                    
                    return status_msg, loaded_list
                    
                except Exception as e:
                    return f"❌ 오류: {str(e)}", "없음"
            
            def clear_documents():
                """문서 초기화"""
                self.rag.clear_documents()
                return "문서가 초기화되었습니다.", "없음"
            
            def chat(message, history):
                """채팅 처리"""
                if history is None:
                    history = []
                
                if not message or not message.strip():
                    return history, ""
                
                loaded_pdfs = self.rag.get_loaded_pdfs()
                if not loaded_pdfs:
                    history.append({"role": "user", "content": message})
                    history.append({"role": "assistant", "content": "먼저 PDF를 업로드해주세요."})
                    return history, ""
                
                try:
                    history.append({"role": "user", "content": message})
                    response = self.rag.query(message)
                    history.append({"role": "assistant", "content": response})
                except Exception as e:
                    history.append({"role": "assistant", "content": f"오류: {str(e)}"})
                
                return history, ""
            
            def change_model(model_name):
                try:
                    self.rag.config.set("model", model_name)
                    self.rag.rag_handler.setup_llm()
                    return f"✅ 모델 변경: {model_name}"
                except Exception as e:
                    return f"❌ 오류: {str(e)}"
            
            def change_temp(temp_value):
                try:
                    self.rag.config.set("temperature", temp_value)
                    self.rag.rag_handler.setup_llm()
                    return f"✅ Temperature: {temp_value}"
                except Exception as e:
                    return f"❌ 오류: {str(e)}"
            
            def save_session(name):
                if name:
                    try:
                        self.rag.chat_handler.save_session(name)
                        return f"✅ 세션 '{name}' 저장됨"
                    except Exception as e:
                        return f"❌ 오류: {str(e)}"
                return "세션 이름을 입력하세요"
            
            def load_session(name):
                if name:
                    try:
                        self.rag.chat_handler.load_session(name)
                        history = []
                        messages = self.rag.chat_handler.get_history()
                        
                        for msg in messages:
                            if hasattr(msg, 'content'):
                                if "Human" in str(type(msg)):
                                    history.append({"role": "user", "content": msg.content})
                                else:
                                    history.append({"role": "assistant", "content": msg.content})
                        
                        return history, f"✅ 세션 '{name}' 로드됨"
                    except Exception as e:
                        return [], f"❌ 오류: {str(e)}"
                return [], "세션 이름을 입력하세요"
            
            def list_sessions():
                sessions_dir = Path.home() / ".rag_gpt" / "sessions"
                result = "세션 이름 | 날짜 | 메시지 수\n"
                result += "-" * 40 + "\n"
                
                if sessions_dir.exists():
                    for session_file in sessions_dir.glob("*.json"):
                        try:
                            with open(session_file, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                            result += f"{data.get('name', 'Unknown')} | "
                            result += f"{data.get('timestamp', '')[:10]} | "
                            result += f"{len(data.get('messages', []))}\n"
                        except:
                            continue
                else:
                    result += "저장된 세션이 없습니다.\n"
                
                return result
            
            def clear_chat():
                return []
            
            def get_loaded_pdfs_display():
                loaded = self.rag.get_loaded_pdfs()
                if loaded:
                    return "\n".join([f"📄 {pdf}" for pdf in loaded])
                return "없음"
            
            # 이벤트 연결
            upload_btn.click(
                process_pdfs, 
                inputs=[pdf_files], 
                outputs=[status, loaded_pdfs]
            )
            clear_docs_btn.click(
                clear_documents,
                inputs=None,
                outputs=[status, loaded_pdfs]
            )
            
            submit.click(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
            msg.submit(chat, inputs=[msg, chatbot], outputs=[chatbot, msg])
            clear.click(clear_chat, inputs=None, outputs=[chatbot])
            
            model_dropdown.change(change_model, inputs=[model_dropdown], outputs=[status])
            temperature.change(change_temp, inputs=[temperature], outputs=[status])
            
            save_btn.click(save_session, inputs=[session_name], outputs=[status])
            load_btn.click(load_session, inputs=[session_name], outputs=[chatbot, status])
            refresh_btn.click(list_sessions, outputs=[sessions_display])
            demo.load(list_sessions, outputs=[sessions_display])
            
        return demo
    
    def launch(self, **kwargs):
        """웹 서버 실행"""
        try:
            demo = self.create_interface()
            print("웹 인터페이스를 시작합니다...")
            print(f"브라우저에서 http://localhost:{kwargs.get('server_port', 7860)} 으로 접속하세요.")
            demo.launch(**kwargs)
        except Exception as e:
            print(f"웹 인터페이스 실행 오류: {e}")
            raise
