"""
Memory Scorer — Bayesian importance scoring with temporal decay.

Importance Score:
    I(f, t) = ω₁·freq(f) + ω₂·recency(f, t) + ω₃·relevance(f)

Where:
    freq(f)     = access_count / max(access_counts)   normalized retrieval frequency
    recency(f)  = e^(-λ · Δt)                        exponential decay (λ=0.01, Δt in hours)
    relevance(f)= base importance score                from initial extraction
    ω           = (0.3, 0.4, 0.3)                    weight vector
"""

import math
from datetime import datetime
from typing import List, Dict
from loguru import logger


class MemoryScorer:
    """
    Computes and updates importance scores for memory fragments.
    Handles temporal decay pruning for stale fragments.
    """

    # Weight vector ω = (freq, recency, relevance)
    WEIGHT_FREQ = 0.3
    WEIGHT_RECENCY = 0.4
    WEIGHT_RELEVANCE = 0.3

    # Decay constant (per hour)
    DECAY_LAMBDA = 0.01

    def __init__(self, sqlite_storage):
        """
        Args:
            sqlite_storage: SQLiteStorageManager instance
        """
        self.sqlite = sqlite_storage

    def compute_importance(self, fragment: Dict, max_access_count: int = 1) -> float:
        """
        Compute importance score for a single fragment.
        
        Args:
            fragment: Dict with fragment_id, access_count, last_accessed, importance_score
            max_access_count: Maximum access count across all fragments (for normalization)
            
        Returns:
            Importance score in [0, 1]
        """
        # Frequency component: normalized access count
        access_count = fragment.get("access_count", 0)
        freq = access_count / max(max_access_count, 1)

        # Recency component: exponential decay
        last_accessed = fragment.get("last_accessed")
        if last_accessed:
            try:
                last_dt = datetime.fromisoformat(last_accessed)
                delta_hours = (datetime.utcnow() - last_dt).total_seconds() / 3600
                recency = math.exp(-self.DECAY_LAMBDA * delta_hours)
            except (ValueError, TypeError):
                recency = 0.5  # Default if parsing fails
        else:
            # Never accessed — use creation time
            created_at = fragment.get("created_at", "")
            try:
                created_dt = datetime.fromisoformat(created_at)
                delta_hours = (datetime.utcnow() - created_dt).total_seconds() / 3600
                recency = math.exp(-self.DECAY_LAMBDA * delta_hours)
            except (ValueError, TypeError):
                recency = 0.5

        # Relevance component: current importance score (from initial extraction)
        relevance = fragment.get("importance_score", 0.5)

        # Weighted sum
        score = (
            self.WEIGHT_FREQ * freq +
            self.WEIGHT_RECENCY * recency +
            self.WEIGHT_RELEVANCE * relevance
        )
        return max(0.0, min(1.0, score))

    def recompute_all(self, session_id: str = None):
        """
        Recompute importance scores for all active fragments (or for a specific session).
        
        Args:
            session_id: Optional — if provided, only recompute for this session
        """
        try:
            if session_id:
                fragments = self.sqlite.get_session_fragments(session_id)
            else:
                # Get all active fragments across all namespaces
                fragments = self.sqlite.get_fragments_by_namespace(["global"])
                # Also include session-scoped fragments
                # (In a full implementation, iterate over all namespaces)

            if not fragments:
                return

            # Find max access count for normalization
            max_access = max(f.get("access_count", 0) for f in fragments)
            max_access = max(max_access, 1)

            for frag in fragments:
                new_score = self.compute_importance(frag, max_access)
                self.sqlite.update_fragment_importance(frag["fragment_id"], new_score)

            logger.info(f"Recomputed importance for {len(fragments)} fragments")

        except Exception as e:
            logger.error(f"Failed to recompute fragment scores: {e}")

    def apply_decay(self, max_age_days: int = 90, min_score: float = 0.1) -> int:
        """
        Soft-delete fragments below the importance threshold after max_age_days.
        
        Returns:
            Number of fragments pruned
        """
        try:
            decayed = self.sqlite.get_decayed_fragments(max_age_days, min_score)
            for frag in decayed:
                self.sqlite.delete_memory_fragment(frag["id"])
            
            if decayed:
                logger.info(f"Decay-pruned {len(decayed)} fragments (age>{max_age_days}d, score<{min_score})")
            return len(decayed)

        except Exception as e:
            logger.error(f"Decay pruning failed: {e}")
            return 0

    def update_on_access(self, fragment_id: str):
        """
        Called when a fragment is retrieved — increment access count
        and update the last_accessed timestamp.
        """
        self.sqlite.update_fragment_access(fragment_id)
