"""
Gradio Frontend for Multimodal RAG System
"""
import gradio as gr
import requests
from typing import List, Tuple, Dict
from loguru import logger

API_BASE = "http://localhost:8000"


class RAGInterface:
    """Gradio interface wrapper"""
    
    def __init__(self):
        self.api_keys = {}
    
    def upload_file(self, file) -> str:
        """Upload and ingest file"""
        if file is None:
            return "No file selected"
        
        try:
            with open(file.name, 'rb') as f:
                files = {'file': (file.name, f)}
                response = requests.post(f"{API_BASE}/ingest/file", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    return f"✓ Ingested {result['atoms_count']} atoms, {result['edges_count']} edges"
                else:
                    return f"Error: {response.text}"
        except Exception as e:
            return f"Error: {str(e)}"
    
    def query(self, question: str, history: List[Dict[str, str]]) -> Tuple:
        """Process query"""
        if not question.strip():
            return history, ""
        
        try:
            # Append user message
            history.append({"role": "user", "content": question})
            
            response = requests.post(
                f"{API_BASE}/query",
                json={"query": question}
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['answer']
            else:
                answer = f"Error: {response.text}"
            
            # Append assistant message
            history.append({"role": "assistant", "content": answer})
            return history, ""
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            history.append({"role": "assistant", "content": f"Error: {str(e)}"})
            return history, ""
    
    def add_api_key(self, provider: str, name: str, key: str, model_name: str = "") -> str:
        """Add API key"""
        if not all([provider, name, key]):
            return "All fields required"
        
        try:
            payload = {
                "provider": provider.lower(),
                "name": name,
                "key": key
            }
            
            if model_name.strip():
                payload["model_name"] = model_name.strip()
            
            response = requests.post(
                f"{API_BASE}/api-keys/add",
                json=payload
            )
            
            if response.status_code == 200:
                # Store locally
                if provider not in self.api_keys:
                    self.api_keys[provider] = []
                self.api_keys[provider].append(name)
                
                msg = f"✓ Added {provider} key: {name}"
                if model_name.strip():
                    msg += f" (Model: {model_name.strip()})"
                return msg
            else:
                return f"Error: {response.text}"
                
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_stats(self) -> str:
        """Get system statistics"""
    
    def get_stats(self) -> str:
        """Get system statistics"""
        try:
            response = requests.get(f"{API_BASE}/statistics")
            if response.status_code == 200:
                stats = response.json()
                
                db_stats = stats.get('database', {})
                llm_stats = stats.get('llm', {})
                weights = stats.get('retrieval_weights', {})
                
                output = "# System Statistics\n\n"
                output += f"**Knowledge Base:**\n"
                output += f"- Total nodes: {db_stats.get('total_nodes', 0)}\n"
                output += f"- Total edges: {db_stats.get('total_edges', 0)}\n\n"
                
                output += f"**LLM Usage:**\n"
                for provider, pstats in llm_stats.items():
                    output += f"- {provider}: {pstats.get('total_requests', 0)} requests\n"
                
                output += f"\n**Retrieval Weights:**\n"
                output += f"- Vector (α): {weights.get('alpha', 0):.3f}\n"
                output += f"- Graph (β): {weights.get('beta', 0):.3f}\n"
                output += f"- BM25 (γ): {weights.get('gamma', 0):.3f}\n"
                
                return output
            else:
                return "Error fetching stats"
        except:
            return "Error connecting to backend"


# Initialize interface
interface = RAGInterface()


# Build UI
with gr.Blocks(title="Multimodal RAG System", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧠 Multimodal RAG System")
    gr.Markdown("Advanced retrieval with vector, graph, and hybrid search")
    
    with gr.Tabs():
        # Chat Tab
        with gr.Tab("💬 Chat"):
            chatbot = gr.Chatbot(height=500)
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask a question...",
                    show_label=False,
                    scale=9
                )
                submit = gr.Button("Send", scale=1)
            
            clear = gr.Button("Clear")
            
            # Event handlers
            submit.click(
                interface.query,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                interface.query,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            )
            
            clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        # Ingest Tab
        with gr.Tab("📁 Ingest Data"):
            gr.Markdown("### Upload Files")
            gr.Markdown("Supported: PDF, DOCX, TXT, Images, Audio, Video, CSV, etc.")
            
            file_upload = gr.File(label="Select File")
            upload_btn = gr.Button("Upload & Process", variant="primary")
            upload_output = gr.Textbox(label="Status", lines=3)
            
            upload_btn.click(
                interface.upload_file,
                inputs=[file_upload],
                outputs=[upload_output]
            )
            
            gr.Markdown("### Ingest URL")
            url_input = gr.Textbox(label="URL", placeholder="https://...")
            url_btn = gr.Button("Ingest URL")
            url_output = gr.Textbox(label="Status")
            
        # API Keys Tab
        with gr.Tab("🔑 API Keys"):
            gr.Markdown("### Add LLM Provider API Keys")
            gr.Markdown("Multiple keys per provider supported for automatic fallback")
            
            with gr.Row():
                provider = gr.Dropdown(
                    choices=["OpenAI", "Anthropic", "Groq", "Gemini", "OpenRouter"],
                    label="Provider"
                )
                key_name = gr.Textbox(label="Key Name", placeholder="My Key 1")
            
            api_key = gr.Textbox(label="API Key", type="password")
            model_name = gr.Textbox(label="Model Name (Optional)", placeholder="e.g. gpt-4, claude-3-opus")
            
            add_key_btn = gr.Button("Add Key", variant="primary")
            key_output = gr.Textbox(label="Status")
            
            add_key_btn.click(
                interface.add_api_key,
                inputs=[provider, key_name, api_key, model_name],
                outputs=[key_output]
            )
            
            gr.Markdown("### Current Keys")
            refresh_keys = gr.Button("Refresh")
            keys_display = gr.Markdown()
        
        # Statistics Tab
        with gr.Tab("📊 Statistics"):
            gr.Markdown("### System Performance")
            
            refresh_stats = gr.Button("Refresh Statistics")
            stats_display = gr.Markdown()
            
            refresh_stats.click(
                interface.get_stats,
                outputs=[stats_display]
            )
    
    # Footer
    gr.Markdown("---")
    gr.Markdown(
        "**Multimodal RAG System** | "
        "Vector + Graph + BM25 Hybrid Retrieval | "
        "Powered by ArangoDB, E5, CLIP, Whisper"
    )


if __name__ == "__main__":
    logger.info("Starting Gradio interface...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )