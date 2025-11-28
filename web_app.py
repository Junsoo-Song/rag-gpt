"""
웹 인터페이스 모듈 (Gradio) - messages 형식
"""
import gradio as gr
from pathlib import Path
from typing import List, Optional
import json

class WebInterface:
    """Gradio 웹 인터페이스"""
    
    def __init__(self, rag_gpt_instance):
        self.rag = rag_gpt_instance
        self.current_pdf = None
        
    def create_interface(self):
        """Gradio 인터페이스 생성"""
        
        with gr.Blocks() as demo:
            gr.Markdown("# 🤖 RAG-GPT: 문서 기반 AI 어시스턴트")
            
            with gr.Tab("💬 대화"):
                with gr.Row():
                    with gr.Column(scale=1):
                        pdf_file = gr.File(
                            label="📄 PDF 업로드",
                            file_types=[".pdf"]
                        )
                        upload_btn = gr.Button("📥 문서 로드", variant="primary")
                        status = gr.Textbox(
                            label="상태",
                            value="문서를 업로드해주세요.",
                            interactive=False
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
                2. **질문하기**: 채팅창에 질문을 입력하고 전송
                3. **세션 저장**: 대화 내용을 저장하려면 '세션 관리' 탭에서 저장
                """)
            
            # 이벤트 핸들러
            def process_pdf(file):
                if file:
                    try:
                        file_path = file.name if hasattr(file, 'name') else file
                        self.rag.load_pdf(Path(file_path))
                        self.current_pdf = file_path
                        return "✅ 문서 로드 완료!"
                    except Exception as e:
                        return f"❌ 오류: {str(e)}"
                return "❌ 파일을 선택해주세요."
            
            def chat(message, history):
                """딕셔너리 형식으로 채팅 처리"""
                if history is None:
                    history = []
                
                if not message or not message.strip():
                    return history, ""
                
                if not self.current_pdf:
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
            
            # 이벤트 연결
            upload_btn.click(process_pdf, inputs=[pdf_file], outputs=[status])
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
