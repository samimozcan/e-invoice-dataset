#!/usr/bin/env python3
"""
Evaluation script for DeepSeek OCR model on Turkish e-invoices.
Computes accuracy metrics for invoice field extraction.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple
from collections import defaultdict

import numpy as np


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    if text is None:
        return ""
    return str(text).strip().lower()


def compute_exact_match(pred: Any, gold: Any) -> float:
    """Compute exact match score."""
    pred_norm = normalize_text(str(pred) if pred else "")
    gold_norm = normalize_text(str(gold) if gold else "")
    
    if gold_norm == "" and pred_norm == "":
        return 1.0  # Both null
    
    return 1.0 if pred_norm == gold_norm else 0.0


def compute_partial_match(pred: str, gold: str) -> float:
    """Compute partial match using token overlap."""
    if pred is None or gold is None:
        return 0.0 if (pred is None) != (gold is None) else 1.0
    
    pred_tokens = set(normalize_text(str(pred)).split())
    gold_tokens = set(normalize_text(str(gold)).split())
    
    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0
    
    intersection = len(pred_tokens & gold_tokens)
    precision = intersection / len(pred_tokens) if pred_tokens else 0
    recall = intersection / len(gold_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)  # F1


def evaluate_header_fields(pred: Dict, gold: Dict) -> Dict[str, Dict]:
    """Evaluate header-level fields."""
    header_fields = [
        "invoice_currency",
        "invoice_customer_address",
        "invoice_customer_country",
        "invoice_customer_name",
        "invoice_date",
        "invoice_delivery_term",
        "invoice_id",
        "invoice_payment_term",
        "invoice_po_number",
        "invoice_shipment_country_of_origin",
        "invoice_supplier_address",
        "invoice_supplier_country",
        "invoice_supplier_name",
        "invoice_supplier_vkn",
        "invoice_total_amount",
        "invoice_total_package_quantity",
        "invoice_total_quantity",
        "invoice_total_gross_weight",
        "invoice_total_net_weight",
    ]
    
    results = {}
    for field in header_fields:
        pred_value = pred.get(field)
        gold_value = gold.get(field)
        
        exact = compute_exact_match(pred_value, gold_value)
        partial = compute_partial_match(
            str(pred_value) if pred_value else None,
            str(gold_value) if gold_value else None
        )
        
        results[field] = {
            "exact_match": exact,
            "partial_match": partial,
            "predicted": pred_value,
            "gold": gold_value,
        }
    
    return results


def evaluate_items(pred_items: List[Dict], gold_items: List[Dict]) -> Dict[str, Any]:
    """Evaluate line items."""
    item_fields = [
        "invoice_item_commodity_code",
        "invoice_item_country_of_origin",
        "invoice_item_description",
        "invoice_item_no",
        "invoice_item_package_quantity",
        "invoice_item_product_id",
        "invoice_item_quantity",
        "invoice_item_total_amount",
        "invoice_item_unit_price",
        "invoice_item_unit_type",
    ]
    
    results = {
        "count_correct": len(pred_items) == len(gold_items),
        "predicted_count": len(pred_items),
        "gold_count": len(gold_items),
        "field_scores": defaultdict(list),
    }
    
    # Match items by invoice_item_no if possible
    pred_by_no = {str(item.get("invoice_item_no", i)): item for i, item in enumerate(pred_items)}
    gold_by_no = {str(item.get("invoice_item_no", i)): item for i, item in enumerate(gold_items)}
    
    all_nos = set(pred_by_no.keys()) | set(gold_by_no.keys())
    
    for item_no in all_nos:
        pred_item = pred_by_no.get(item_no, {})
        gold_item = gold_by_no.get(item_no, {})
        
        for field in item_fields:
            pred_value = pred_item.get(field)
            gold_value = gold_item.get(field)
            
            exact = compute_exact_match(pred_value, gold_value)
            partial = compute_partial_match(
                str(pred_value) if pred_value else None,
                str(gold_value) if gold_value else None
            )
            
            results["field_scores"][field].append({
                "exact_match": exact,
                "partial_match": partial,
            })
    
    # Aggregate scores
    results["field_averages"] = {}
    for field, scores in results["field_scores"].items():
        results["field_averages"][field] = {
            "exact_match": np.mean([s["exact_match"] for s in scores]) if scores else 0,
            "partial_match": np.mean([s["partial_match"] for s in scores]) if scores else 0,
        }
    
    return results


def evaluate_sample(pred: Dict, gold: Dict) -> Dict[str, Any]:
    """Evaluate a single prediction against ground truth."""
    # Header fields
    header_results = evaluate_header_fields(pred, gold)
    
    # Items
    pred_items = pred.get("items", [])
    gold_items = gold.get("items", [])
    item_results = evaluate_items(pred_items, gold_items)
    
    # Aggregate scores
    header_exact = np.mean([r["exact_match"] for r in header_results.values()])
    header_partial = np.mean([r["partial_match"] for r in header_results.values()])
    
    item_exact = np.mean([
        r["exact_match"] for r in item_results["field_averages"].values()
    ]) if item_results["field_averages"] else 0
    
    item_partial = np.mean([
        r["partial_match"] for r in item_results["field_averages"].values()
    ]) if item_results["field_averages"] else 0
    
    return {
        "header": header_results,
        "items": item_results,
        "summary": {
            "header_exact_match": header_exact,
            "header_partial_match": header_partial,
            "items_exact_match": item_exact,
            "items_partial_match": item_partial,
            "overall_exact_match": (header_exact + item_exact) / 2,
            "overall_partial_match": (header_partial + item_partial) / 2,
        }
    }


def evaluate_dataset(
    predictions_file: str,
    ground_truth_file: str,
) -> Dict[str, Any]:
    """Evaluate predictions against ground truth for entire dataset."""
    
    # Load predictions
    with open(predictions_file, 'r', encoding='utf-8') as f:
        if predictions_file.endswith('.jsonl'):
            predictions = [json.loads(line) for line in f if line.strip()]
        else:
            predictions = json.load(f)
    
    # Load ground truth
    with open(ground_truth_file, 'r', encoding='utf-8') as f:
        if ground_truth_file.endswith('.jsonl'):
            ground_truth = [json.loads(line) for line in f if line.strip()]
        else:
            ground_truth = json.load(f)
    
    # Match by image path or index
    results = []
    
    if isinstance(predictions, list) and isinstance(ground_truth, list):
        for pred, gold in zip(predictions, ground_truth):
            # Handle nested data structure
            if "conversations" in gold:
                # Extract gold from conversation
                for conv in gold["conversations"]:
                    if conv.get("role") == "<|Assistant|>":
                        gold_data = json.loads(conv["content"])
                        break
                else:
                    gold_data = {}
            else:
                gold_data = gold
            
            pred_data = pred if not pred.get("error") else {}
            
            result = evaluate_sample(pred_data, gold_data)
            result["source"] = pred.get("source_file", gold.get("image", "unknown"))
            results.append(result)
    
    # Aggregate results
    all_summaries = [r["summary"] for r in results]
    
    aggregate = {
        "num_samples": len(results),
        "header_exact_match": np.mean([s["header_exact_match"] for s in all_summaries]),
        "header_partial_match": np.mean([s["header_partial_match"] for s in all_summaries]),
        "items_exact_match": np.mean([s["items_exact_match"] for s in all_summaries]),
        "items_partial_match": np.mean([s["items_partial_match"] for s in all_summaries]),
        "overall_exact_match": np.mean([s["overall_exact_match"] for s in all_summaries]),
        "overall_partial_match": np.mean([s["overall_partial_match"] for s in all_summaries]),
    }
    
    return {
        "aggregate": aggregate,
        "per_sample": results,
    }


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a formatted way."""
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    
    agg = results["aggregate"]
    
    print(f"\nNumber of samples: {agg['num_samples']}")
    print("\n--- Header Fields ---")
    print(f"  Exact Match:   {agg['header_exact_match']:.2%}")
    print(f"  Partial Match: {agg['header_partial_match']:.2%}")
    
    print("\n--- Line Items ---")
    print(f"  Exact Match:   {agg['items_exact_match']:.2%}")
    print(f"  Partial Match: {agg['items_partial_match']:.2%}")
    
    print("\n--- Overall ---")
    print(f"  Exact Match:   {agg['overall_exact_match']:.2%}")
    print(f"  Partial Match: {agg['overall_partial_match']:.2%}")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate invoice extraction model")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to predictions file (JSON or JSONL)")
    parser.add_argument("--ground-truth", type=str, required=True,
                        help="Path to ground truth file (JSON or JSONL)")
    parser.add_argument("--output", type=str,
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    results = evaluate_dataset(args.predictions, args.ground_truth)
    
    print_results(results)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
