"""AI Interpretation Module for TerraSentinel AI.

Supports optional integration with officially supported Meta AI / Llama 3 OpenAI-compatible
inference endpoints (e.g., vLLM, Together AI, Groq, Fireworks, Ollama, Azure, or AWS Bedrock).
Falls back cleanly to Local Analysis Mode if no endpoint or key is configured.
"""
import os
import json
import logging
from typing import Dict, Any, Optional
import requests
from config import META_AI_CONFIG

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are TerraSentinel AI's scientific remote sensing analyst.
Your objective is to provide an evidence-based assessment of land-cover changes detected
from satellite optical imagery.

CRITICAL SCIENTIFIC PRINCIPLES:
1. Distinguish strictly between detected visual changes and inferred real-world causes.
2. An image alone NEVER establishes human intent, deliberate destruction, or illegality.
3. Explicitly report environmental alternatives (e.g. seasonal phenology, agricultural cycles, sun angle).
4. Never invent geographic coordinates, satellite telemetry, or false confidence.
5. Return ONLY a valid JSON object matching the requested schema.
"""

def generate_ai_interpretation(
    metrics: Dict[str, Any],
    heuristic_results: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Generates an evidence-based interpretation. If external AI API is available

    and configured, queries the endpoint. Otherwise, wraps local heuristic output
    in a standardized scientific structure with explicit 'Local Analysis Mode' status.
    """
    api_endpoint = META_AI_CONFIG.get("api_endpoint", "").strip()
    api_key = META_AI_CONFIG.get("api_key", "").strip()
    model_name = META_AI_CONFIG.get("model", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo")

    # If no API configured, return local analysis mode
    if not api_endpoint or not api_key or "your_meta_ai_api_key_here" in api_key:
        return _format_local_response(heuristic_results, metrics, metadata)

    # Prepare evidence payload for the LLM
    evidence_payload = {
        "metadata": {
            "location": metadata.get("location_name") or "Unspecified location",
            "date_before": metadata.get("date_before") or "Unspecified date",
            "date_after": metadata.get("date_after") or "Unspecified date",
            "sensor": metadata.get("sensor_name") or "Optical RGB sensor",
            "registration_status": metadata.get("registration_status", "Default grid alignment"),
            "resolution_m_px": metadata.get("approx_resolution_m", "Uncalibrated optical resolution")
        },
        "computed_metrics": {
            "total_pixels_monitored": metrics.get("total_image_pixels", 0),
            "changed_pixels": metrics.get("total_changed_pixels", 0),
            "change_percentage": metrics.get("change_percentage", 0.0),
            "region_count": metrics.get("regions_count", 0),
            "largest_region_pixels": metrics.get("largest_region_pixels", 0),
            "largest_region_pct": metrics.get("largest_region_percentage", 0.0),
            "average_intensity": metrics.get("average_change_intensity", 0.0)
        },
        "heuristic_spectral_indicators": heuristic_results.get("heuristic_indicators", {}),
        "heuristic_primary_category": heuristic_results.get("primary_category"),
        "heuristic_confidence": heuristic_results.get("confidence")
    }

    user_prompt = f"""Analyze the following satellite image comparison evidence and return a structured JSON assessment.

COMPUTED SATELLITE EVIDENCE:
{json.dumps(evidence_payload, indent=2)}

Return a JSON object with EXACTLY these keys:
{{
  "summary": "Evidence-based summary string",
  "observed_changes": ["observation 1", "observation 2"],
  "possible_categories": ["category 1", "category 2"],
  "confidence": "low, moderate, or high",
  "confidence_basis": "Scientific explanation of supporting evidence",
  "alternative_explanations": ["alternative 1", "alternative 2"],
  "recommended_follow_up": ["action 1", "action 2"],
  "limitations": ["limitation 1", "limitation 2"]
}}
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"} if "together" in api_endpoint or "openai" in api_endpoint else None
    }

    try:
        response = requests.post(
            api_endpoint,
            headers=headers,
            json={k: v for k, v in payload.items() if v is not None},
            timeout=META_AI_CONFIG.get("timeout", 15)
        )

        if response.status_code == 200:
            res_json = response.json()
            content = res_json["choices"][0]["message"]["content"]
            parsed = json.loads(content)

            # Validate mandatory schema keys
            required_keys = [
                "summary", "observed_changes", "possible_categories",
                "confidence", "confidence_basis", "alternative_explanations",
                "recommended_follow_up", "limitations"
            ]
            if all(k in parsed for k in required_keys):
                parsed["mode"] = f"Meta AI Assisted ({model_name})"
                return parsed

        logger.warning(f"Meta AI returned non-200 code or incomplete schema: {response.status_code}. Using local fallback.")
    except Exception as e:
        logger.info(f"External AI endpoint query skipped or timed out: {e}. Falling back to local mode.")

    # Safe fallback if API query encountered any issue
    return _format_local_response(heuristic_results, metrics, metadata)


def _format_local_response(
    heuristic: Dict[str, Any],
    metrics: Dict[str, Any],
    metadata: Dict[str, Any]
) -> Dict[str, Any]:
    """Wraps the deterministic local computer-vision evaluation into the standard schema."""
    primary_cat = heuristic.get("primary_category", "Unclassified visual change")
    total_px = metrics.get("total_changed_pixels", 0)
    pct = metrics.get("change_percentage", 0.0)

    summary = (
        f"Local deterministic computer-vision analysis detected {metrics.get('regions_count', 0)} contiguous "
        f"change clusters ({total_px:,} pixels, {pct:.2f}% coverage). Visual indicators are most consistent "
        f"with '{primary_cat}' at a {heuristic.get('confidence', 'Moderate').lower()} confidence level."
    )

    return {
        "mode": "Local Analysis Mode (Deterministic Computer Vision)",
        "summary": summary,
        "observed_changes": [
            heuristic.get("observation", f"Detected {total_px} pixels of surface variation."),
            f"Mean change intensity across detected pixels: {metrics.get('average_change_intensity', 0)} DN.",
            f"Registration status: {metadata.get('registration_status', 'Default grid alignment')}."
        ],
        "possible_categories": heuristic.get("possible_categories", [primary_cat]),
        "confidence": heuristic.get("confidence", "Moderate").lower(),
        "confidence_basis": heuristic.get("confidence_basis", "Evaluated from spectral transitions and cluster geometries."),
        "alternative_explanations": heuristic.get("alternative_explanations", []),
        "recommended_follow_up": heuristic.get("recommended_follow_up", []),
        "limitations": heuristic.get("limitations", [])
    }
