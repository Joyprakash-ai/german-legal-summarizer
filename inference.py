#!/usr/bin/env python3
"""
Inference script for German Legal Summarizer.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import Dict, Optional
import argparse
import PyPDF2
import re

class LegalSummarizer:
    """Main class for legal document summarization."""
    
    def __init__(self, model_path: str = "./legal-summarizer-model"):
        """Initialize the summarizer with model and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def preprocess_text(self, text: str, language: str = "de") -> str:
        """Preprocess legal text for summarization."""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Language-specific preprocessing
        if language == "de":
            # German-specific cleanup
            text = re.sub(r'§\s*(\d+)', r'§\1', text)  # Fix law paragraph formatting
            text = re.sub(r'Abs\.\s*(\d+)', r'Abs.\1', text)  # Fix paragraph formatting
        elif language == "en":
            # English-specific cleanup
            text = re.sub(r'Sec\.\s*(\d+)', r'Section \1', text)
        
        return text
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file."""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            print(f"Error reading PDF: {e}")
        return text
    
    def summarize(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 30,
        language: str = "de",
        num_beams: int = 4,
        temperature: float = 0.7
    ) -> Dict[str, str]:
        """
        Summarize legal text.
        
        Args:
            text: The legal text to summarize
            max_length: Maximum summary length
            min_length: Minimum summary length
            language: Language code ('de' for German, 'en' for English)
            num_beams: Number of beams for beam search
            temperature: Sampling temperature
        
        Returns:
            Dictionary with summary and metadata
        """
        # Preprocess text
        processed_text = self.preprocess_text(text, language)
        
        # Tokenize input
        inputs = self.tokenizer(
            processed_text,
            max_length=1024,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        ).to(self.device)
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                min_length=min_length,
                num_beams=num_beams,
                temperature=temperature,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        # Decode summary
        summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Post-process summary
        summary = self.postprocess_summary(summary, language)
        
        return {
            "summary": summary,
            "language": language,
            "original_length": len(processed_text),
            "summary_length": len(summary),
            "compression_ratio": f"{len(summary) / len(processed_text) * 100:.1f}%"
        }
    
    def postprocess_summary(self, summary: str, language: str) -> str:
        """Post-process generated summary."""
        # Capitalize first letter
        if summary and summary[0].islower():
            summary = summary[0].upper() + summary[1:]
        
        # Ensure summary ends with period
        if summary and not summary.endswith(('.', '!', '?')):
            summary = summary + '.'
        
        # Language-specific post-processing
        if language == "de":
            # Ensure proper noun capitalization in German
            lines = summary.split('. ')
            processed_lines = []
            for line in lines:
                words = line.split()
                if words:
                    # Capitalize nouns (simplified heuristic)
                    for i, word in enumerate(words):
                        if i > 0 and word[0].isupper() and len(word) > 3:
                            continue  # Probably already a noun
                processed_lines.append(' '.join(words))
            summary = '. '.join(processed_lines)
        
        return summary

def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description="German Legal Document Summarizer")
    parser.add_argument("--text", type=str, help="Text to summarize")
    parser.add_argument("--file", type=str, help="File to summarize (PDF or TXT)")
    parser.add_argument("--language", type=str, default="de", choices=["de", "en"],
                       help="Language of input text")
    parser.add_argument("--model", type=str, default="./legal-summarizer-model",
                       help="Path to model directory")
    parser.add_argument("--max_length", type=int, default=150,
                       help="Maximum summary length")
    parser.add_argument("--output", type=str, help="Output file for results")
    
    args = parser.parse_args()
    
    # Initialize summarizer
    summarizer = LegalSummarizer(model_path=args.model)
    
    # Get text from input
    text = ""
    if args.text:
        text = args.text
    elif args.file:
        if args.file.lower().endswith('.pdf'):
            text = summarizer.extract_text_from_pdf(args.file)
        else:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
    else:
        print("Error: Please provide either --text or --file")
        return
    
    if not text.strip():
        print("Error: No text to summarize")
        return
    
    # Generate summary
    print(f"Summarizing {len(text)} characters...")
    result = summarizer.summarize(
        text=text,
        max_length=args.max_length,
        language=args.language
    )
    
    # Output results
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    print(result["summary"])
    print("\n" + "="*50)
    print("METADATA:")
    print(f"Language: {result['language']}")
    print(f"Original length: {result['original_length']} characters")
    print(f"Summary length: {result['summary_length']} characters")
    print(f"Compression ratio: {result['compression_ratio']}")
    
    # Save to file if requested
    if args.output:
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()
