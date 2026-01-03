# German Legal Document Summarizer ⚖️📄

A fine-tuned language model specialized in summarizing complex German legal documents into clear, concise executive summaries. Built for lawyers, paralegals, and businesses dealing with German contracts and regulations.

**✨ Features**
- **Legal-Specific Fine-Tuning:** Model trained on German legal texts (contracts, terms, regulations).
- **Bilingual Summaries:** Outputs summaries in both German and English.
- **Context-Aware:** Maintains key legal clauses, dates, parties, and obligations.
- **API & Web Interface:** REST API for integration and Gradio UI for easy use.
- **Customizable:** Can be fine-tuned further on your specific legal domain.

**🎯 Use Case & Impact**
> **Problem:** A law firm spends hours reviewing standard Mietverträge (rental contracts) for clients.
> **Solution:** This tool summarizes a 20-page contract in seconds, highlighting key clauses and risks.
> **Goal:** Reduce legal review time by 70% for routine documents.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- GPU recommended for training, CPU okay for inference
- Hugging Face account (for model hub)

### Installation
```bash
# 1. Clone repository
git clone https://github.com/joyprakash-ai/german-legal-summarizer.git
cd german-legal-summarizer

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download pre-trained model
python download_model.py
