#!/usr/bin/env python3
"""
RAG Research Agent - Generates comprehensive PDF reports on specific research topics
using data from the RAG system.
"""
import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import ollama
import torch
from transformers import AutoModel, AutoTokenizer


# Mean-pool token embeddings with attention mask to get a sentence embedding.
def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1.0)
    return summed / counts


# Embed a query string using the local transformer model.
def embed_query(query: str, model_path: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        outputs = model(**inputs)
        pooled = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
    return pooled.cpu().numpy()[0]


# Compute cosine similarity between a matrix of vectors and a single vector.
def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return np.dot(a_norm, b_norm)


# Call the local Ollama server with a prompt and return the response text.
def call_ollama(prompt: str, model: str, base_url: str) -> str:
    client = ollama.Client(host=base_url)
    response = client.generate(model=model, prompt=prompt, stream=False)
    return response.get("response", "")


def extract_country(source_pdf: str | None) -> str:
    if not source_pdf:
        return "unknown"
    match = re.search(r"NU_JWC_(?:ETI_)?([A-Z]+)", str(source_pdf).upper())
    if match:
        return match.group(1)
    return Path(source_pdf).stem if source_pdf else "unknown"


def research_topic(
    research_query: str,
    json_path: Path,
    vectors_path: Path,
    model_path: str,
    ollama_model: str,
    max_chunks: int = 20,
    score_threshold: float = 0.7,
) -> Dict:
    """Research a topic using RAG and return comprehensive results."""
    print(f"Researching topic: {research_query}")
    
    # Load chunks
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    chunks = payload.get("chunks", [])
    
    # Load vectors
    vectors = np.load(vectors_path, allow_pickle=False)["vectors"]
    if len(chunks) != vectors.shape[0]:
        raise ValueError("Vector count does not match chunk count.")
    
    # Embed query
    print("Embedding query...")
    query_vec = embed_query(research_query, model_path)
    scores = cosine_sim(vectors, query_vec)
    sorted_idx = np.argsort(scores)[::-1]
    
    # Get relevant chunks
    best_score = float(scores[sorted_idx[0]])
    threshold = max(score_threshold, best_score * 0.85)
    eligible_idx = [i for i in sorted_idx if float(scores[i]) >= threshold]
    context_idx = eligible_idx[:max_chunks]
    
    print(f"Found {len(context_idx)} relevant chunks (best score: {best_score:.3f})")
    
    # Collect context
    context_chunks = []
    countries = set()
    for idx in context_idx:
        chunk = chunks[int(idx)]
        countries.add(extract_country(chunk.get("source_pdf")))
        context_chunks.append({
            "chunk": chunk,
            "score": float(scores[idx]),
            "country": extract_country(chunk.get("source_pdf")),
        })
    
    # Generate comprehensive research report using LLM
    print("Generating research report...")
    context_text = "\n\n".join(
        f"[Source: {item['country']}, Score: {item['score']:.3f}]\n"
        f"{item['chunk']['text']}"
        for item in context_chunks
    )
    
    research_prompt = (
        "You are a NATO analyst tasked with creating a comprehensive research report. "
        "Analyze the provided context and create a detailed, well-structured report on the research topic.\n\n"
        "Research Topic: {query}\n\n"
        "Context from Documents:\n{context}\n\n"
        "Create a comprehensive report with the following structure:\n"
        "1. Executive Summary (2-3 paragraphs)\n"
        "2. Key Findings (detailed analysis)\n"
        "3. Country-Specific Information (if applicable)\n"
        "4. Strategic Implications\n"
        "5. Recommendations\n"
        "6. Sources and References\n\n"
        "Use only the information provided in the context. If information is insufficient, state so explicitly. "
        "Be thorough, accurate, and professional in your analysis."
    ).format(query=research_query, context=context_text)
    
    report_content = call_ollama(research_prompt, ollama_model, "http://localhost:11434")
    
    return {
        "research_query": research_query,
        "report_content": report_content,
        "context_chunks": context_chunks,
        "countries": sorted(list(countries)),
        "best_score": best_score,
        "chunks_used": len(context_chunks),
        "generated_at": datetime.now().isoformat(),
    }


def create_pdf_report(research_data: Dict, output_path: Path) -> None:
    """Create a PDF report from research data."""
    print(f"Creating PDF report: {output_path}")
    
    with PdfPages(output_path) as pdf:
        # Title page
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        title_text = f"Research Report\n{research_data['research_query']}"
        ax.text(0.5, 0.7, title_text, ha='center', va='center', 
                fontsize=20, fontweight='bold', wrap=True)
        ax.text(0.5, 0.4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.3, f"Sources: {len(research_data['countries'])} countries, "
                f"{research_data['chunks_used']} document chunks", 
                ha='center', va='center', fontsize=10)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Main report content
        report_text = research_data['report_content']
        paragraphs = report_text.split('\n\n')
        
        current_page_text = []
        current_height = 0
        max_height = 10.5  # Leave margins
        
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_xlim(0, 8.5)
        ax.set_ylim(0, 11)
        
        y_position = 10.5
        line_height = 0.25
        font_size = 10
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if it's a heading
            is_heading = (para.startswith('#') or 
                         para.isupper() and len(para) < 100 or
                         re.match(r'^\d+\.\s+[A-Z]', para))
            
            if is_heading:
                if current_height > max_height * 0.7:  # New page for headings
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
                    fig = plt.figure(figsize=(8.5, 11))
                    ax = fig.add_subplot(111)
                    ax.axis('off')
                    ax.set_xlim(0, 8.5)
                    ax.set_ylim(0, 11)
                    y_position = 10.5
                
                # Draw heading (wrap if too long)
                heading_text = re.sub(r'^#+\s*', '', para)
                if len(heading_text) > 85:
                    # Split long headings
                    words = heading_text.split()
                    heading_lines = []
                    current = []
                    for word in words:
                        if len(' '.join(current + [word])) > 85 and current:
                            heading_lines.append(' '.join(current))
                            current = [word]
                        else:
                            current.append(word)
                    if current:
                        heading_lines.append(' '.join(current))
                    for hline in heading_lines:
                        if y_position < 0.75:
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close()
                            fig = plt.figure(figsize=(8.5, 11))
                            ax = fig.add_subplot(111)
                            ax.axis('off')
                            ax.set_xlim(0, 8.5)
                            ax.set_ylim(0, 11)
                            y_position = 10.5
                        ax.text(0.75, y_position, hline, ha='left', va='top',
                               fontsize=14, fontweight='bold')
                        y_position -= 0.3
                    y_position -= 0.2
                else:
                    ax.text(0.75, y_position, heading_text, ha='left', va='top',
                           fontsize=14, fontweight='bold')
                    y_position -= 0.5
                y_position -= 0.5
            else:
                # Wrap text manually (approximately 85 characters per line)
                words = para.split()
                lines = []
                current_line = []
                max_chars = 85
                
                for word in words:
                    test_line = ' '.join(current_line + [word])
                    if len(test_line) > max_chars and current_line:
                        lines.append(' '.join(current_line))
                        current_line = [word]
                    else:
                        current_line.append(word)
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                # Draw paragraph
                for line in lines:
                    if y_position < 0.5:  # New page
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
                        fig = plt.figure(figsize=(8.5, 11))
                        ax = fig.add_subplot(111)
                        ax.axis('off')
                        ax.set_xlim(0, 8.5)
                        ax.set_ylim(0, 11)
                        y_position = 10.5
                    
                    ax.text(0.75, y_position, line, ha='left', va='top',
                           fontsize=font_size)
                    y_position -= line_height
                
                y_position -= 0.1  # Paragraph spacing
        
        # Save last page
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # Sources page
        fig = plt.figure(figsize=(8.5, 11))
        ax = fig.add_subplot(111)
        ax.axis('off')
        ax.set_xlim(0, 8.5)
        ax.set_ylim(0, 11)
        
        ax.text(0.5, 10.5, "Sources and References", ha='left', va='top',
               fontsize=16, fontweight='bold', transform=ax.transAxes)
        
        y_pos = 9.5
        ax.text(0.5, y_pos, f"Countries Referenced: {', '.join(research_data['countries'])}", 
               ha='left', va='top', fontsize=11, transform=ax.transAxes)
        y_pos -= 0.5
        
        ax.text(0.5, y_pos, f"Document Chunks Used: {research_data['chunks_used']}", 
               ha='left', va='top', fontsize=11, transform=ax.transAxes)
        y_pos -= 0.5
        
        ax.text(0.5, y_pos, f"Best Match Score: {research_data['best_score']:.3f}", 
               ha='left', va='top', fontsize=11, transform=ax.transAxes)
        y_pos -= 1.0
        
        ax.text(0.5, y_pos, "Top Sources:", ha='left', va='top',
               fontsize=12, fontweight='bold', transform=ax.transAxes)
        y_pos -= 0.4
        
        for i, item in enumerate(research_data['context_chunks'][:10], 1):
            source_text = (
                f"{i}. {item['country']} (Score: {item['score']:.3f}) - "
                f"Chunk {item['chunk'].get('chunk_id', 'N/A')}, "
                f"Pages {item['chunk'].get('page_start', 'N/A')}-{item['chunk'].get('page_end', 'N/A')}"
            )
            if y_pos < 1.0:
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                fig = plt.figure(figsize=(8.5, 11))
                ax = fig.add_subplot(111)
                ax.axis('off')
                ax.set_xlim(0, 8.5)
                ax.set_ylim(0, 11)
                y_pos = 10.5
            
            # Wrap long source text
            if len(source_text) > 100:
                words = source_text.split()
                source_lines = []
                current = []
                for word in words:
                    if len(' '.join(current + [word])) > 100 and current:
                        source_lines.append(' '.join(current))
                        current = [word]
                    else:
                        current.append(word)
                if current:
                    source_lines.append(' '.join(current))
                for sline in source_lines:
                    if y_pos < 1.0:
                        pdf.savefig(fig, bbox_inches='tight')
                        plt.close()
                        fig = plt.figure(figsize=(8.5, 11))
                        ax = fig.add_subplot(111)
                        ax.axis('off')
                        ax.set_xlim(0, 8.5)
                        ax.set_ylim(0, 11)
                        y_pos = 10.5
                    ax.text(0.5, y_pos, sline, ha='left', va='top',
                           fontsize=9, transform=ax.transAxes)
                    y_pos -= 0.25
            else:
                ax.text(0.5, y_pos, source_text, ha='left', va='top',
                       fontsize=9, transform=ax.transAxes)
                y_pos -= 0.3
            y_pos -= 0.3
        
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    print(f"PDF report saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate PDF research reports using RAG system."
    )
    parser.add_argument(
        "research_query",
        help="Research topic or question to investigate.",
    )
    parser.add_argument(
        "--json",
        default="cms.json",
        help="JSON file with chunks (default: cms.json).",
    )
    parser.add_argument(
        "--vectors",
        default="cms.npz",
        help="Vectors file (default: cms.npz).",
    )
    parser.add_argument(
        "--model-path",
        default="/home/linkages/cursor/pdftext/models/mxbai-embed-large-v1",
        help="Local embedding model path.",
    )
    parser.add_argument(
        "--ollama-model",
        default="llama3.2",
        help="Ollama model name (default: llama3.2).",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output PDF file path (default: research_report_<timestamp>.pdf).",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=20,
        help="Maximum chunks to use for research (default: 20).",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.7,
        help="Minimum score threshold for chunks (default: 0.7).",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    vectors_path = Path(args.vectors)
    
    if not json_path.exists():
        print(f"Error: JSON file not found: {json_path}")
        return
    
    if not vectors_path.exists():
        print(f"Error: Vectors file not found: {vectors_path}")
        return

    # Conduct research
    research_data = research_topic(
        args.research_query,
        json_path,
        vectors_path,
        args.model_path,
        args.ollama_model,
        args.max_chunks,
        args.score_threshold,
    )
    
    # Generate PDF
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_query = re.sub(r'[^\w\s-]', '', args.research_query)[:30]
        safe_query = re.sub(r'[-\s]+', '_', safe_query)
        output_path = Path(f"research_report_{safe_query}_{timestamp}.pdf")
    
    create_pdf_report(research_data, output_path)
    
    print(f"\nResearch complete! Report saved to: {output_path}")


if __name__ == "__main__":
    main()
