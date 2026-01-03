"""
Gradio web interface for German Legal Summarizer.
"""

import gradio as gr
from inference import LegalSummarizer
import tempfile
import os

# Initialize the summarizer
summarizer = LegalSummarizer()

def summarize_legal_text(text, language, length_slider):
    """Summarize legal text from input box."""
    if not text.strip():
        return "Please enter legal text to summarize.", "", ""
    
    try:
        result = summarizer.summarize(
            text=text,
            max_length=length_slider,
            language=language
        )
        
        # Format output
        output = f"""**Summary ({language.upper()}):**
{result['summary']}

**Details:**
- Original: {result['original_length']} characters
- Summary: {result['summary_length']} characters
- Compression: {result['compression_ratio']}"""
        
        return output, result['summary'], text
    
    except Exception as e:
        return f"Error: {str(e)}", "", ""

def summarize_file(file, language, length_slider):
    """Summarize legal text from uploaded file."""
    if file is None:
        return "Please upload a file.", "", ""
    
    # Read file content
    try:
        with open(file.name, 'r', encoding='utf-8') as f:
            text = f.read()
    except:
        # Try different encoding
        with open(file.name, 'r', encoding='latin-1') as f:
            text = f.read()
    
    return summarize_legal_text(text, language, length_slider)

def create_interface():
    """Create Gradio interface."""
    with gr.Blocks(title="German Legal Summarizer", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# ⚖️ German Legal Document Summarizer")
        gr.Markdown("Summarize complex German legal documents into clear, concise summaries.")
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input methods tabs
                with gr.Tabs():
                    with gr.TabItem("📝 Paste Text"):
                        text_input = gr.Textbox(
                            label="Legal Text",
                            placeholder="Paste German legal text here...",
                            lines=10
                        )
                        text_button = gr.Button("Summarize Text", variant="primary")
                    
                    with gr.TabItem("📁 Upload File"):
                        file_input = gr.File(label="Upload Document", type="file")
                        file_button = gr.Button("Summarize File", variant="primary")
                
                # Parameters
                with gr.Accordion("⚙️ Settings", open=False):
                    language_radio = gr.Radio(
                        ["de", "en"],
                        label="Text Language",
                        value="de",
                        info="German (de) or English (en)"
                    )
                    length_slider = gr.Slider(
                        minimum=50,
                        maximum=300,
                        value=150,
                        step=10,
                        label="Summary Length"
                    )
            
            with gr.Column(scale=3):
                # Outputs
                output_markdown = gr.Markdown(label="Results")
                
                with gr.Accordion("📋 Extracted Text", open=False):
                    original_text = gr.Textbox(label="Original Text", lines=5)
                
                with gr.Accordion("📄 Clean Summary", open=False):
                    clean_summary = gr.Textbox(label="Summary Only", lines=3)
                
                # Example section
                with gr.Accordion("📚 Examples", open=False):
                    gr.Markdown("""
                    **Try these examples:**
                    
                    **German Contract Clause:**
                    ```
                    Der Mieter verpflichtet sich, die Miete monatlich im Voraus bis zum dritten Werktag des Monats auf das vom Vermieter bezeichnete Konto zu überweisen. Bei Verzug zahlt der Mieter Verzugszinsen in Höhe von 5 Prozentpunkten über dem Basiszinssatz. Die Miete beträgt 850 Euro warm, inklusive aller Nebenkosten mit Ausnahme von Strom und Internet.
                    ```
                    
                    **German Legal Text:**
                    ```
                    Gemäß § 535 BGB ist der Vermieter verpflichtet, die Mietsache dem Mieter in einem zum vertragsgemäßen Gebrauch geeigneten Zustand zu überlassen und sie während der Mietzeit in diesem Zustand zu erhalten. Der Mieter ist gemäß § 536c BGB verpflichtet, dem Vermieter Mängel der Mietsache unverzüglich anzuzeigen.
                    ```
                    """)
        
        # Footer
        gr.Markdown("---")
        gr.Markdown("""
        **How it works:**
        1. Paste German legal text or upload a document
        2. Adjust settings if needed
        3. Click "Summarize" to generate a concise summary
        4. Use the summary for quick review or translation
        
        **Note:** This tool is for assistance only. Always consult a qualified lawyer for legal advice.
        """)
        
        # Connect buttons
        text_button.click(
            fn=summarize_legal_text,
            inputs=[text_input, language_radio, length_slider],
            outputs=[output_markdown, clean_summary, original_text]
        )
        
        file_button.click(
            fn=summarize_file,
            inputs=[file_input, language_radio, length_slider],
            outputs=[output_markdown, clean_summary, original_text]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False
    )
