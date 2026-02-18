"""
Research Context Graph — Purpose-built graph schema for deep research.
Provides infinite context management via a structured networkx graph
with provenance tracking (Decision Traces).
"""

import uuid
import json
from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime
from loguru import logger
import networkx as nx

from backend.core.deep_research.models import (
    ResearchNodeType, ResearchEdgeType,
    Finding, SourceInfo
)


class ResearchContextGraph:
    """
    In-memory directed graph for storing and traversing research context.
    
    Implements the Context Graph paradigm:
    - Nodes represent research artifacts (sessions, plans, steps, sources, findings, syntheses)
    - Edges represent relationships (provenance, extraction, support/contradiction)
    - Full decision trace from final synthesis → findings → sources
    
    Each deep research session gets its own graph instance.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.graph = nx.DiGraph()
        self._node_index: Dict[str, Dict] = {}  # id → node attrs (fast lookup)
        self._type_index: Dict[str, List[str]] = {}  # type → [node_ids]
        
        logger.info(f"[ContextGraph] Initialized for session {session_id}")

    # ═══════════════════════════════════════════════════
    # Node Operations
    # ═══════════════════════════════════════════════════

    def add_node(
        self,
        node_type: ResearchNodeType,
        properties: Dict[str, Any],
        node_id: Optional[str] = None
    ) -> str:
        """Add a node to the graph. Returns the node ID."""
        nid = node_id or str(uuid.uuid4())
        
        attrs = {
            "id": nid,
            "type": node_type.value,
            "created_at": datetime.utcnow().isoformat(),
            **properties
        }
        
        self.graph.add_node(nid, **attrs)
        self._node_index[nid] = attrs
        
        # Update type index
        type_key = node_type.value
        if type_key not in self._type_index:
            self._type_index[type_key] = []
        self._type_index[type_key].append(nid)
        
        return nid

    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node attributes by ID."""
        return self._node_index.get(node_id)

    def update_node(self, node_id: str, updates: Dict[str, Any]):
        """Update node properties."""
        if node_id in self._node_index:
            self._node_index[node_id].update(updates)
            nx.set_node_attributes(self.graph, {node_id: updates})

    def get_nodes_by_type(self, node_type: ResearchNodeType) -> List[Dict]:
        """Get all nodes of a specific type."""
        type_key = node_type.value
        ids = self._type_index.get(type_key, [])
        return [self._node_index[nid] for nid in ids if nid in self._node_index]

    # ═══════════════════════════════════════════════════
    # Edge Operations
    # ═══════════════════════════════════════════════════

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: ResearchEdgeType,
        properties: Optional[Dict] = None
    ):
        """Add a directed edge between two nodes."""
        attrs = {
            "type": edge_type.value,
            "created_at": datetime.utcnow().isoformat(),
            **(properties or {})
        }
        self.graph.add_edge(source_id, target_id, **attrs)

    def get_edges_from(self, node_id: str, edge_type: Optional[ResearchEdgeType] = None) -> List[Tuple[str, str, Dict]]:
        """Get all outgoing edges from a node, optionally filtered by type."""
        edges = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if edge_type is None or data.get("type") == edge_type.value:
                edges.append((node_id, target, data))
        return edges

    def get_edges_to(self, node_id: str, edge_type: Optional[ResearchEdgeType] = None) -> List[Tuple[str, str, Dict]]:
        """Get all incoming edges to a node, optionally filtered by type."""
        edges = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if edge_type is None or data.get("type") == edge_type.value:
                edges.append((source, node_id, data))
        return edges

    # ═══════════════════════════════════════════════════
    # High-Level Queries
    # ═══════════════════════════════════════════════════

    def add_finding(
        self,
        finding: Finding,
        source_node_id: str,
        step_node_id: Optional[str] = None
    ) -> str:
        """Add a finding node and link it to its source."""
        finding_id = finding.id or str(uuid.uuid4())
        finding.id = finding_id
        
        node_id = self.add_node(
            ResearchNodeType.FINDING,
            {
                "content": finding.content,
                "source_url": finding.source_url,
                "source_title": finding.source_title,
                "extraction_type": finding.extraction_type,
                "importance": finding.importance,
                "depth": finding.depth,
            },
            node_id=finding_id
        )
        
        # Link finding → source (provenance)
        self.add_edge(node_id, source_node_id, ResearchEdgeType.EXTRACTED_FROM)
        
        # Link finding → step
        if step_node_id:
            self.add_edge(node_id, step_node_id, ResearchEdgeType.BELONGS_TO)
        
        return node_id

    def add_source(self, url: str, title: str, domain: str, content_length: int = 0) -> str:
        """Add a web source node, or return existing if URL matches."""
        # Check for existing source with same URL
        for nid in self._type_index.get(ResearchNodeType.WEB_SOURCE.value, []):
            if self._node_index.get(nid, {}).get("url") == url:
                return nid

        return self.add_node(
            ResearchNodeType.WEB_SOURCE,
            {
                "url": url,
                "title": title,
                "domain": domain,
                "content_length": content_length,
                "scraped_at": datetime.utcnow().isoformat(),
            }
        )

    def add_synthesis(self, content: str, finding_ids: List[str], depth: int = 0, parent_synthesis_id: Optional[str] = None) -> str:
        """Add a synthesis node and link it to its input findings or parent synthesis."""
        synth_id = self.add_node(
            ResearchNodeType.SYNTHESIS,
            {
                "content": content,
                "depth": depth,
                "input_count": len(finding_ids),
            }
        )
        
        # Link synthesis → each input finding
        for fid in finding_ids:
            self.add_edge(synth_id, fid, ResearchEdgeType.SYNTHESIZED_FROM)
        
        # Link to parent synthesis (recursive chain)
        if parent_synthesis_id:
            self.add_edge(synth_id, parent_synthesis_id, ResearchEdgeType.DERIVED_FROM)
        
        return synth_id

    def get_findings_for_step(self, step_id: str) -> List[Dict]:
        """Get all findings that belong to a research step."""
        findings = []
        for finding_node in self.get_nodes_by_type(ResearchNodeType.FINDING):
            edges = self.get_edges_from(finding_node["id"], ResearchEdgeType.BELONGS_TO)
            for _, target, _ in edges:
                if target == step_id:
                    findings.append(finding_node)
                    break
        return findings

    def get_all_findings(self) -> List[Dict]:
        """Get all findings across all steps."""
        return self.get_nodes_by_type(ResearchNodeType.FINDING)

    def get_all_sources(self) -> List[SourceInfo]:
        """Get all web sources with their finding counts."""
        sources = []
        for source_node in self.get_nodes_by_type(ResearchNodeType.WEB_SOURCE):
            # Count findings extracted from this source
            incoming = self.get_edges_to(source_node["id"], ResearchEdgeType.EXTRACTED_FROM)
            sources.append(SourceInfo(
                url=source_node.get("url", ""),
                title=source_node.get("title", ""),
                domain=source_node.get("domain", ""),
                relevance_score=0.0,
                findings_count=len(incoming)
            ))
        return sources

    def get_source_count(self) -> int:
        """Count unique web sources."""
        return len(self._type_index.get(ResearchNodeType.WEB_SOURCE.value, []))

    def get_findings_count(self) -> int:
        """Count total findings."""
        return len(self._type_index.get(ResearchNodeType.FINDING.value, []))

    def get_synthesis_chain(self) -> List[Dict]:
        """Get the synthesis chain (ordered by depth)."""
        syntheses = self.get_nodes_by_type(ResearchNodeType.SYNTHESIS)
        return sorted(syntheses, key=lambda n: n.get("depth", 0))

    # ═══════════════════════════════════════════════════
    # Decision Trace (Provenance)
    # ═══════════════════════════════════════════════════

    def get_decision_trace(self) -> List[Dict]:
        """
        Produce a full provenance trace from final synthesis → findings → sources.
        This is the Decision Trace described in the Context Graph architecture.
        """
        trace = []
        syntheses = self.get_synthesis_chain()
        
        if not syntheses:
            return trace
        
        # Start from the highest-depth synthesis (final report)
        final = syntheses[-1] if syntheses else None
        if not final:
            return trace
        
        trace.append({
            "type": "final_synthesis",
            "content_preview": final.get("content", "")[:200],
            "depth": final.get("depth", 0),
            "input_count": final.get("input_count", 0),
        })
        
        # Traverse down through synthesized_from edges
        visited = set()
        stack = [final["id"]]
        
        while stack:
            current_id = stack.pop()
            if current_id in visited:
                continue
            visited.add(current_id)
            
            edges = self.get_edges_from(current_id, ResearchEdgeType.SYNTHESIZED_FROM)
            for _, target_id, _ in edges:
                target_node = self.get_node(target_id)
                if target_node:
                    node_type = target_node.get("type")
                    if node_type == ResearchNodeType.FINDING.value:
                        # Trace finding to its source
                        source_edges = self.get_edges_from(target_id, ResearchEdgeType.EXTRACTED_FROM)
                        source_info = None
                        for _, src_id, _ in source_edges:
                            src_node = self.get_node(src_id)
                            if src_node:
                                source_info = {
                                    "url": src_node.get("url", ""),
                                    "title": src_node.get("title", ""),
                                }
                                break
                        
                        trace.append({
                            "type": "finding",
                            "content_preview": target_node.get("content", "")[:150],
                            "importance": target_node.get("importance", 0.5),
                            "source": source_info,
                        })
                    elif node_type == ResearchNodeType.SYNTHESIS.value:
                        trace.append({
                            "type": "intermediate_synthesis",
                            "depth": target_node.get("depth", 0),
                            "input_count": target_node.get("input_count", 0),
                        })
                    stack.append(target_id)
        
        return trace

    # ═══════════════════════════════════════════════════
    # Serialization
    # ═══════════════════════════════════════════════════

    def export_graph(self) -> str:
        """Serialize the graph to JSON string for persistence."""
        data = {
            "session_id": self.session_id,
            "nodes": [],
            "edges": []
        }
        
        for nid, attrs in self.graph.nodes(data=True):
            data["nodes"].append({"id": nid, **attrs})
        
        for src, tgt, attrs in self.graph.edges(data=True):
            data["edges"].append({"source": src, "target": tgt, **attrs})
        
        return json.dumps(data, default=str)

    @classmethod
    def import_graph(cls, json_str: str) -> "ResearchContextGraph":
        """Deserialize a graph from JSON string."""
        data = json.loads(json_str)
        graph = cls(data["session_id"])
        
        for node in data["nodes"]:
            nid = node.pop("id")
            node_type = node.get("type", "finding")
            # Reconstruct using raw graph addition to preserve all attrs
            graph.graph.add_node(nid, id=nid, **node)
            graph._node_index[nid] = {"id": nid, **node}
            
            if node_type not in graph._type_index:
                graph._type_index[node_type] = []
            graph._type_index[node_type].append(nid)
        
        for edge in data["edges"]:
            src = edge.pop("source")
            tgt = edge.pop("target")
            graph.graph.add_edge(src, tgt, **edge)
        
        return graph

    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "sources": self.get_source_count(),
            "findings": self.get_findings_count(),
            "syntheses": len(self._type_index.get(ResearchNodeType.SYNTHESIS.value, [])),
            "search_queries": len(self._type_index.get(ResearchNodeType.SEARCH_QUERY.value, [])),
        }
