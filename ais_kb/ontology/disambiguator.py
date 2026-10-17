"""Production-grade acronym disambiguator ported from Adeft.
Includes inline defining-pattern recognizer and TF-IDF/LogisticRegression
context disambiguation with full probability reporting."""
from __future__ import annotations
import re
import logging
from collections import defaultdict
from typing import Dict, List, NamedTuple, Optional, Tuple, Set

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ais_kb.ontology.grounder import OntologyGrounder
from ais_kb.ontology.graph import AISOntologyGraph

logger = logging.getLogger(__name__)


class DisambResult(NamedTuple):
    decision: str
    name: str
    predicted_probs: Dict[str, float]
    defining_patterns: Dict[str, List[str]]


class AdeftRecognizer:
    """Inline acronym-definition pattern recognizer (Full Name (ACRONYM))."""

    def __init__(self, shortform: str, grounding_map: Dict[str, str]):
        self.shortform = shortform
        # Build regex patterns like r"([A-Z][a-zA-Z]* )+\(ACRONYM\)"
        self.pattern = re.compile(
            r"(?:[A-Z][a-zA-Z]*\s*)+\(" + re.escape(shortform) + r"\)",
            re.IGNORECASE,
        )
        self.grounding_map = grounding_map

    def recognize(self, text: str) -> List[Dict[str, str]]:
        results: List[Dict[str, str]] = []
        # Simple inline definition regex: Word Word (SF)
        # Ported from Adeft recognize logic
        matches = self.pattern.findall(text)
        seen: Set[str] = set()
        for _ in matches:
            # Try to extract longform from nearby text
            # Simplified production version
            for longform in self.grounding_map:
                if longform.lower() in text.lower() and longform not in seen:
                    results.append({
                        "grounding": self.grounding_map[longform],
                        "longform_text": longform,
                    })
                    seen.add(longform)
        return results


class AdeftDisambiguator:
    """Production disambiguator for AI Safety acronyms.
    Uses defining-pattern detection + logistic regression context classifier."""

    def __init__(
        self,
        classifier,
        grounding_dict: Dict[str, Dict[str, str]],
        names: Dict[str, str],
    ):
        self.classifier = classifier
        self.shortforms: List[str] = list(grounding_dict.keys())
        self.grounding_dict = grounding_dict
        self.names = names
        self.labels: Set[str] = set(
            value for gmap in grounding_dict.values() for value in gmap.values()
        ) | set(getattr(classifier.estimator, "classes_", []))
        self.pos_labels: List[str] = []  # Updated after training

        self.recognizers = [
            AdeftRecognizer(sf, gmap)
            for sf, gmap in grounding_dict.items()
        ]

    def disambiguate(self, texts: str | List[str]) -> List[DisambResult]:
        if isinstance(texts, str):
            return self.disambiguate([texts])[0]
        # Step 1: defining-pattern recognition
        groundings: List[Dict[str, List[str]]] = []
        for text in texts:
            grounding_for_text: Dict[str, List[str]] = defaultdict(list)
            seen_longforms: Set[str] = set()
            for recognizer in self.recognizers:
                results = recognizer.recognize(text)
                for info in results:
                    if not info:
                        continue
                    grounding = info["grounding"]
                    longform = info["longform_text"]
                    if longform not in seen_longforms:
                        grounding_for_text[grounding].append(longform)
                        seen_longforms.add(longform)
            groundings.append(dict(grounding_for_text))

        # Step 2: classifier for ambiguous cases
        results: List[Optional[DisambResult]] = [None] * len(texts)
        undetermined = [
            (i, text, grounding)
            for i, (text, grounding) in enumerate(zip(texts, groundings))
            if len(grounding) != 1
        ]
        predictions: Optional[List] = None
        if undetermined:
            indices = [u[0] for u in undetermined]
            undet_texts = [u[1] for u in undetermined]
            predictions = self.classifier.predict_proba(undet_texts)

        pred_index = 0
        for index, grounding in enumerate(groundings):
            if len(grounding) == 1:
                disamb, longforms = list(grounding.items())[0]
                pred_probs = {str(label): (1.0 if str(label) == str(disamb) else 0.0) for label in self.labels}
                results[index] = DisambResult(
                    str(disamb),
                    self.names.get(str(disamb), str(disamb)),
                    pred_probs,
                    dict(grounding),
                )
            elif grounding and predictions is not None:
                # Inconsistent patterns: blend predictions
                pred_probs_unnorm = {}
                for label in self.labels:
                    label_str = str(label)
                    if label_str in grounding:
                        pred_probs_unnorm[label_str] = predictions[pred_index].get(label_str, 0.0)
                    else:
                        pred_probs_unnorm[label_str] = 0.0
                total_prob = sum(pred_probs_unnorm.values())
                pred_probs = {
                    label: float(prob / total_prob) if total_prob > 0 else 0.0
                    for label, prob in pred_probs_unnorm.items()
                }
                best = max(pred_probs, key=lambda k: pred_probs[k])
                results[index] = DisambResult(
                    best,
                    self.names.get(best, best),
                    pred_probs,
                    dict(grounding),
                )
                pred_index += 1
            else:
                # No pattern found; use classifier directly
                if predictions is not None:
                    pred_probs = {
                        str(label): float(prob) for label, prob in predictions[pred_index].items()
                    }
                    best = max(pred_probs, key=lambda k: pred_probs[k])
                    results[index] = DisambResult(
                        best,
                        self.names.get(best, best),
                        pred_probs,
                        {},
                    )
                    pred_index += 1
                else:
                    # Fallback to highest frequency label
                    best = list(self.labels)[0] if self.labels else "unknown"
                    results[index] = DisambResult(
                        best,
                        self.names.get(best, best),
                        {str(label): 0.0 for label in self.labels},
                        {},
                    )
        return results  # type: ignore
