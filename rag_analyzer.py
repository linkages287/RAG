#!/usr/bin/env python3
"""
RAG Analysis Agent - Generates comprehensive reports on RAG system performance.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_chat_history(chat_log_path: Path) -> List[Dict]:
    """Load chat history from a JSON log file."""
    if not chat_log_path.exists():
        return []
    with chat_log_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def analyze_query_distribution(messages: List[Dict]) -> Dict:
    """Analyze distribution of query types."""
    query_types = [msg.get("query_type", "unknown") for msg in messages if msg.get("role") == "user"]
    type_counts = Counter(query_types)
    total = len(query_types)
    
    return {
        "total_queries": total,
        "rag_queries": type_counts.get("rag", 0),
        "general_queries": type_counts.get("general", 0),
        "rag_percentage": (type_counts.get("rag", 0) / total * 100) if total > 0 else 0,
        "general_percentage": (type_counts.get("general", 0) / total * 100) if total > 0 else 0,
    }


def analyze_score_distribution(messages: List[Dict]) -> Dict:
    """Analyze score distribution for RAG queries."""
    scores = []
    low_score_queries = []
    
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("query_type") == "rag":
            top_results = msg.get("top_results", [])
            if top_results:
                best_score = top_results[0].get("score", 0)
                scores.append(best_score)
                if best_score < 0.6:
                    # Find corresponding user query
                    idx = messages.index(msg)
                    if idx > 0 and messages[idx - 1].get("role") == "user":
                        low_score_queries.append({
                            "query": messages[idx - 1].get("content", ""),
                            "score": best_score,
                        })
    
    if not scores:
        return {
            "avg_score": 0,
            "min_score": 0,
            "max_score": 0,
            "median_score": 0,
            "queries_below_0.6": len(low_score_queries),
            "low_score_examples": low_score_queries[:5],
        }
    
    return {
        "avg_score": float(np.mean(scores)),
        "min_score": float(np.min(scores)),
        "max_score": float(np.max(scores)),
        "median_score": float(np.median(scores)),
        "std_score": float(np.std(scores)),
        "queries_below_0.6": len(low_score_queries),
        "queries_below_0.6_percentage": (len(low_score_queries) / len(scores) * 100) if scores else 0,
        "low_score_examples": low_score_queries[:5],
    }


def analyze_context_usage(messages: List[Dict]) -> Dict:
    """Analyze how context chunks are being used."""
    context_counts = []
    country_usage = Counter()
    
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("query_type") == "rag":
            top_results = msg.get("top_results", [])
            if top_results:
                context_counts.append(len(top_results))
                for result in top_results:
                    country = result.get("country", "unknown")
                    country_usage[country] += 1
    
    return {
        "avg_context_chunks": float(np.mean(context_counts)) if context_counts else 0,
        "min_context_chunks": int(np.min(context_counts)) if context_counts else 0,
        "max_context_chunks": int(np.max(context_counts)) if context_counts else 0,
        "most_used_countries": dict(country_usage.most_common(10)),
    }


def analyze_response_quality(messages: List[Dict]) -> Dict:
    """Analyze response quality indicators."""
    response_lengths = []
    error_count = 0
    
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            response_lengths.append(len(content))
            if "error" in content.lower() or "LLM error" in content:
                error_count += 1
    
    return {
        "avg_response_length": float(np.mean(response_lengths)) if response_lengths else 0,
        "min_response_length": int(np.min(response_lengths)) if response_lengths else 0,
        "max_response_length": int(np.max(response_lengths)) if response_lengths else 0,
        "error_count": error_count,
        "error_percentage": (error_count / len(response_lengths) * 100) if response_lengths else 0,
    }


def generate_text_report(analysis: Dict) -> str:
    """Generate a human-readable text report."""
    report = []
    report.append("=" * 80)
    report.append("RAG SYSTEM ANALYSIS REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Query Distribution
    report.append("QUERY DISTRIBUTION")
    report.append("-" * 80)
    qd = analysis["query_distribution"]
    report.append(f"Total Queries: {qd['total_queries']}")
    report.append(f"RAG Queries: {qd['rag_queries']} ({qd['rag_percentage']:.1f}%)")
    report.append(f"General Knowledge Queries: {qd['general_queries']} ({qd['general_percentage']:.1f}%)")
    report.append("")
    
    # Score Distribution
    report.append("SCORE DISTRIBUTION (RAG Queries)")
    report.append("-" * 80)
    sd = analysis["score_distribution"]
    report.append(f"Average Score: {sd['avg_score']:.3f}")
    report.append(f"Median Score: {sd['median_score']:.3f}")
    report.append(f"Min Score: {sd['min_score']:.3f}")
    report.append(f"Max Score: {sd['max_score']:.3f}")
    if "std_score" in sd:
        report.append(f"Standard Deviation: {sd['std_score']:.3f}")
    report.append(f"Queries Below 0.6 Threshold: {sd['queries_below_0.6']} ({sd.get('queries_below_0.6_percentage', 0):.1f}%)")
    if sd.get("low_score_examples"):
        report.append("\nLow Score Query Examples:")
        for ex in sd["low_score_examples"]:
            report.append(f"  - Score {ex['score']:.3f}: {ex['query'][:60]}...")
    report.append("")
    
    # Context Usage
    report.append("CONTEXT USAGE")
    report.append("-" * 80)
    cu = analysis["context_usage"]
    report.append(f"Average Context Chunks: {cu['avg_context_chunks']:.1f}")
    report.append(f"Min Context Chunks: {cu['min_context_chunks']}")
    report.append(f"Max Context Chunks: {cu['max_context_chunks']}")
    if cu.get("most_used_countries"):
        report.append("\nMost Referenced Countries:")
        for country, count in cu["most_used_countries"].items():
            report.append(f"  - {country}: {count} times")
    report.append("")
    
    # Response Quality
    report.append("RESPONSE QUALITY")
    report.append("-" * 80)
    rq = analysis["response_quality"]
    report.append(f"Average Response Length: {rq['avg_response_length']:.0f} characters")
    report.append(f"Min Response Length: {rq['min_response_length']} characters")
    report.append(f"Max Response Length: {rq['max_response_length']} characters")
    report.append(f"Errors: {rq['error_count']} ({rq['error_percentage']:.1f}%)")
    report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 80)
    recommendations = []
    
    if sd.get("queries_below_0.6_percentage", 0) > 20:
        recommendations.append(
            f"⚠️  High percentage ({sd['queries_below_0.6_percentage']:.1f}%) of queries below 0.6 threshold. "
            "Consider improving document coverage or query formulation."
        )
    
    if sd.get("avg_score", 0) < 0.7:
        recommendations.append(
            f"⚠️  Average score ({sd['avg_score']:.3f}) is relatively low. "
            "Consider tuning the embedding model or improving document quality."
        )
    
    if rq.get("error_percentage", 0) > 5:
        recommendations.append(
            f"⚠️  Error rate ({rq['error_percentage']:.1f}%) is high. "
            "Check LLM connectivity and model availability."
        )
    
    if qd.get("rag_percentage", 0) < 50:
        recommendations.append(
            f"ℹ️  Only {qd['rag_percentage']:.1f}% of queries use RAG. "
            "Most queries are general knowledge - consider if document coverage is sufficient."
        )
    
    if not recommendations:
        recommendations.append("✓ System performance looks good. No major issues detected.")
    
    for rec in recommendations:
        report.append(rec)
    
    report.append("")
    report.append("=" * 80)
    
    return "\n".join(report)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze RAG system performance from chat history."
    )
    parser.add_argument(
        "chat_log",
        help="Path to chat history JSON file (or session data).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="rag_analysis_report.txt",
        help="Output file for text report (default: rag_analysis_report.txt).",
    )
    parser.add_argument(
        "--json-output",
        "-j",
        help="Optional JSON output file for structured data.",
    )
    args = parser.parse_args()

    chat_log_path = Path(args.chat_log)
    if not chat_log_path.exists():
        print(f"Error: Chat log file not found: {chat_log_path}")
        return

    print(f"Loading chat history from {chat_log_path}...")
    messages = load_chat_history(chat_log_path)
    
    if not messages:
        print("No messages found in chat history.")
        return

    print(f"Analyzing {len(messages)} messages...")
    
    # Perform analysis
    analysis = {
        "query_distribution": analyze_query_distribution(messages),
        "score_distribution": analyze_score_distribution(messages),
        "context_usage": analyze_context_usage(messages),
        "response_quality": analyze_response_quality(messages),
        "total_messages": len(messages),
        "analysis_timestamp": datetime.now().isoformat(),
    }
    
    # Generate text report
    text_report = generate_text_report(analysis)
    
    # Save text report
    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(text_report)
    print(f"\nText report saved to: {output_path}")
    print("\n" + text_report)
    
    # Save JSON if requested
    if args.json_output:
        json_path = Path(args.json_output)
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        print(f"\nJSON data saved to: {json_path}")


if __name__ == "__main__":
    main()
