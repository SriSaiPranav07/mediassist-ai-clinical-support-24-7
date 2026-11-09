"""
MediAssist - Medical Guideline RAG (Retrieval-Augmented Generation) Engine
Retrieves authoritative clinical practice guidelines and synthesizes evidence-grounded recommendations.
Author: Zelvora Industry Labs / MediAssist Team
"""
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MedicalGuidelineRAGEngine:
    """
    Retrieval-Augmented Generation (RAG) engine for clinical decision support.
    Grounds AI findings in peer-reviewed clinical guidelines (AHA/ACC, ATS/IDSA, ADA, BTS, CAP/Tietz).
    """

    def __init__(self, guidelines_path: Optional[Path] = None):
        self.guidelines_path = guidelines_path or (Path(__file__).resolve().parent.parent / "data" / "medical_guidelines.json")
        self.guidelines: List[Dict[str, Any]] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.corpus_matrix = None
        self._load_corpus()

    def _load_corpus(self):
        """Loads and vectorizes the clinical guideline knowledge base."""
        if not self.guidelines_path.exists():
            return
        
        try:
            with open(self.guidelines_path, "r", encoding="utf-8") as f:
                self.guidelines = json.load(f)
            
            # Construct rich search document for each guideline
            docs = []
            for g in self.guidelines:
                keywords_str = " ".join(g.get("keywords", []))
                doc_text = f"{g.get('topic', '')} {keywords_str} {g.get('summary', '')} {g.get('recommendation', '')}"
                docs.append(doc_text)
            
            if docs:
                self.vectorizer = TfidfVectorizer(
                    ngram_range=(1, 2),
                    stop_words="english",
                    sublinear_tf=True
                )
                self.corpus_matrix = self.vectorizer.fit_transform(docs)
        except Exception:
            self.guidelines = []
            self.vectorizer = None
            self.corpus_matrix = None

    def retrieve(self, query: str, top_k: int = 2, min_relevance: float = 0.08) -> List[Dict[str, Any]]:
        """
        Retrieves top-k most clinically relevant medical guidelines for a given query or symptom presentation.
        """
        if not self.guidelines or self.vectorizer is None or self.corpus_matrix is None or not query:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.corpus_matrix)[0]
            
            # Boost scores for explicit keyword matches
            query_lower = query.lower()
            boosted_scores = []
            for idx, g in enumerate(self.guidelines):
                score = float(similarities[idx])
                for kw in g.get("keywords", []):
                    if kw.lower() in query_lower:
                        score += 0.15
                boosted_scores.append(score)
            
            ranked_indices = np.argsort(boosted_scores)[::-1]
            results = []
            
            for idx in ranked_indices[:top_k]:
                score = float(boosted_scores[idx])
                if score >= min_relevance:
                    g_copy = dict(self.guidelines[idx])
                    g_copy["relevance_score"] = round(min(score, 1.0), 3)
                    results.append(g_copy)
            
            return results
        except Exception:
            return []

    def synthesize_guideline_evidence(
        self,
        presentation_text: str,
        urgency_category: str,
        detected_pathologies: Optional[List[str]] = None,
        critical_labs: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Synthesizes grounded clinical insights from retrieved guideline evidence.
        """
        search_terms = [presentation_text]
        if detected_pathologies:
            search_terms.extend(detected_pathologies)
        if critical_labs:
            search_terms.extend(critical_labs)
        
        full_query = " ".join(search_terms)
        retrieved_docs = self.retrieve(full_query, top_k=2)
        
        if not retrieved_docs:
            return {
                "has_evidence": False,
                "guidelines": [],
                "synthesized_summary": "No specific guideline indexed for current presentation parameters.",
                "action_recommendation": "Follow institutional clinical protocols and standard clinical judgment."
            }
        
        top_guideline = retrieved_docs[0]
        summary_text = (
            f"Findings are supported by {top_guideline['source']} ({top_guideline['year']}). "
            f"Standard practice emphasizes: {top_guideline['summary'][:220]}..."
        )
        recommendation = top_guideline.get("recommendation", "Maintain close clinical monitoring.")
        
        return {
            "has_evidence": True,
            "guidelines": retrieved_docs,
            "synthesized_summary": summary_text,
            "action_recommendation": recommendation,
            "primary_source": f"{top_guideline['source']} ({top_guideline['year']})"
        }
