import numpy as np
from arango import ArangoClient
from arango.database import StandardDatabase
from arango.collection import StandardCollection
from arango.graph import Graph
from typing import List, Dict, Optional, Tuple
import json
from loguru import logger

from backend.models.config import (
    DatabaseConfig, KnowledgeAtom, GraphEdge, 
    EdgeType, Modality, Resolution
)


class ArangoStorageManager:
    """
    Unified storage manager for vector + graph storage using ArangoDB
    Implements hybrid vector-graph database functionality
    """
    
    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.client: Optional[ArangoClient] = None
        self.db: Optional[StandardDatabase] = None
        self.graph: Optional[Graph] = None
        self.nodes_collection: Optional[StandardCollection] = None
        self.edges_collection: Optional[StandardCollection] = None
        
    def connect(self):
        """Establish connection to ArangoDB"""
        try:
            self.client = ArangoClient(
                hosts=f"http://{self.config.host}:{self.config.port}"
            )
            
            # Connect to system database
            sys_db = self.client.db(
                '_system',
                username=self.config.username,
                password=self.config.password
            )
            
            # Create database if not exists
            if not sys_db.has_database(self.config.database):
                sys_db.create_database(self.config.database)
                logger.info(f"Created database: {self.config.database}")
            
            # Connect to our database
            self.db = self.client.db(
                self.config.database,
                username=self.config.username,
                password=self.config.password
            )
            
            # Initialize collections and graph
            self._initialize_collections()
            self._initialize_graph()
            self._create_indexes()
            
            logger.info("Successfully connected to ArangoDB")
            
        except Exception as e:
            logger.error(f"Failed to connect to ArangoDB: {e}")
            raise
    
    def _initialize_collections(self):
        """Create collections if they don't exist"""
        # Nodes collection (vertex collection)
        if not self.db.has_collection(self.config.nodes_collection):
            self.nodes_collection = self.db.create_collection(
                self.config.nodes_collection
            )
            logger.info(f"Created collection: {self.config.nodes_collection}")
        else:
            self.nodes_collection = self.db.collection(self.config.nodes_collection)
        
        # Edges collection
        if not self.db.has_collection(self.config.edges_collection):
            self.edges_collection = self.db.create_collection(
                self.config.edges_collection,
                edge=True
            )
            logger.info(f"Created edge collection: {self.config.edges_collection}")
        else:
            self.edges_collection = self.db.collection(self.config.edges_collection)

        # System configuration collection
        if not self.db.has_collection("system_config"):
            self.system_config = self.db.create_collection("system_config")
            logger.info("Created collection: system_config")
        else:
            self.system_config = self.db.collection("system_config")
    
    def _initialize_graph(self):
        """Create graph if it doesn't exist"""
        if not self.db.has_graph(self.config.graph_name):
            self.graph = self.db.create_graph(self.config.graph_name)
            
            # Add vertex collection
            if not self.graph.has_vertex_collection(self.config.nodes_collection):
                self.graph.create_vertex_collection(self.config.nodes_collection)
            
            # Add edge definition
            if not self.graph.has_edge_definition(self.config.edges_collection):
                self.graph.create_edge_definition(
                    edge_collection=self.config.edges_collection,
                    from_vertex_collections=[self.config.nodes_collection],
                    to_vertex_collections=[self.config.nodes_collection]
                )
            
            logger.info(f"Created graph: {self.config.graph_name}")
        else:
            self.graph = self.db.graph(self.config.graph_name)
    
    def _create_indexes(self):
        """Create necessary indexes for performance"""
        # Persistent indexes for metadata
        self.nodes_collection.add_persistent_index(
            fields=['modality', 'resolution'],
            name='modality_resolution_idx'
        )
        
        self.nodes_collection.add_persistent_index(
            fields=['timestamp'],
            name='timestamp_idx'
        )
        
        self.nodes_collection.add_persistent_index(
            fields=['source'],
            name='source_idx'
        )
        
        # Full-text index for content
        self.nodes_collection.add_fulltext_index(
            fields=['content'],
            name='content_fulltext_idx'
        )
        
        # Edge type index
        self.edges_collection.add_persistent_index(
            fields=['edge_type'],
            name='edge_type_idx'
        )
        
        self.edges_collection.add_persistent_index(
            fields=['weight'],
            name='weight_idx'
        )
        
        logger.info("Created database indexes")
    
    def insert_knowledge_atom(self, atom: KnowledgeAtom) -> str:
        """
        Insert a knowledge atom into the graph
        Returns the node_id
        """
        document = {
            'content': atom.content,
            'modality': atom.modality.value,
            'resolution': atom.resolution.value,
            'embeddings': atom.embeddings,
            'metadata': atom.metadata,
            'source': atom.source,
            'timestamp': atom.timestamp,
            'entities': atom.entities
        }
        
        result = self.nodes_collection.insert(document)
        node_id = result['_key']
        
        logger.debug(f"Inserted knowledge atom: {node_id}")
        return node_id
    
    def insert_edge(self, edge: GraphEdge) -> str:
        """Insert an edge into the graph"""
        document = {
            '_from': f"{self.config.nodes_collection}/{edge.source_id}",
            '_to': f"{self.config.nodes_collection}/{edge.target_id}",
            'edge_type': edge.edge_type.value,
            'weight': edge.weight,
            'metadata': edge.metadata
        }
        
        result = self.edges_collection.insert(document)
        return result['_key']
    
    def batch_insert_atoms(self, atoms: List[KnowledgeAtom]) -> List[str]:
        """Batch insert knowledge atoms"""
        documents = [
            {
                'content': atom.content,
                'modality': atom.modality.value,
                'resolution': atom.resolution.value,
                'embeddings': atom.embeddings,
                'metadata': atom.metadata,
                'source': atom.source,
                'timestamp': atom.timestamp,
                'entities': atom.entities
            }
            for atom in atoms
        ]
        
        results = self.nodes_collection.insert_many(documents)
        node_ids = [r['_key'] for r in results]
        
        logger.info(f"Batch inserted {len(node_ids)} knowledge atoms")
        return node_ids
    
    def batch_insert_edges(self, edges: List[GraphEdge]):
        """Batch insert edges"""
        documents = [
            {
                '_from': f"{self.config.nodes_collection}/{edge.source_id}",
                '_to': f"{self.config.nodes_collection}/{edge.target_id}",
                'edge_type': edge.edge_type.value,
                'weight': edge.weight,
                'metadata': edge.metadata
            }
            for edge in edges
        ]
        
        self.edges_collection.insert_many(documents)
        logger.info(f"Batch inserted {len(edges)} edges")
    
    def vector_search(
        self, 
        query_embedding: np.ndarray,
        modality: Optional[Modality] = None,
        resolution: Optional[Resolution] = None,
        top_k: int = 20,
        embedding_key: str = "default"
    ) -> List[Tuple[str, float]]:
        """
        Vector similarity search using cosine similarity
        Returns list of (node_id, similarity_score)
        """
        # Build filter
        filter_parts = []
        if modality:
            filter_parts.append(f"doc.modality == '{modality.value}'")
        if resolution:
            filter_parts.append(f"doc.resolution == '{resolution.value}'")
        
        filter_clause = " AND " + " AND ".join(filter_parts) if filter_parts else ""
        
        # AQL query for vector similarity
        query = f"""
        FOR doc IN {self.config.nodes_collection}
            {f"FILTER {' AND '.join(filter_parts)}" if filter_parts else ""}
            LET embedding = doc.embeddings['{embedding_key}']
            FILTER embedding != null
            LET similarity = (
                SUM(
                    FOR i IN 0..LENGTH(embedding)-1
                        RETURN embedding[i] * @query_vec[i]
                ) / (
                    SQRT(SUM(FOR x IN embedding RETURN x * x)) *
                    SQRT(SUM(FOR x IN @query_vec RETURN x * x))
                )
            )
            SORT similarity DESC
            LIMIT @top_k
            RETURN {{
                node_id: doc._key,
                similarity: similarity,
                content: doc.content,
                modality: doc.modality,
                resolution: doc.resolution
            }}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                'query_vec': query_embedding.tolist(),
                'top_k': top_k
            }
        )
        
        results = [(r['node_id'], r['similarity']) for r in cursor]
        return results
    
    def graph_traversal(
        self,
        start_nodes: List[str],
        max_depth: int = 2,
        edge_types: Optional[List[EdgeType]] = None,
        limit: int = 100  # Add result limit for performance
    ) -> List[Dict]:
        """
        Optimized graph traversal from seed nodes.
        Uses LIMIT and PRUNE for performance on dense graphs.
        """
        edge_filter = ""
        if edge_types:
            types_str = ", ".join([f"'{et.value}'" for et in edge_types])
            edge_filter = f"FILTER e.edge_type IN [{types_str}]"
        
        # Convert start_nodes to full document IDs
        start_docs = [f"{self.config.nodes_collection}/{nid}" for nid in start_nodes[:10]]  # Limit seeds
        
        # Optimized query with LIMIT and PRUNE
        query = f"""
        LET results = (
            FOR start IN @start_nodes
                FOR v, e, p IN 1..@max_depth ANY start {self.config.edges_collection}
                    PRUNE LENGTH(p.edges) >= @max_depth
                    {edge_filter}
                    LIMIT @limit
                    RETURN DISTINCT {{
                        node_id: v._key,
                        content: SUBSTRING(v.content, 0, 500),
                        modality: v.modality,
                        distance: LENGTH(p.edges),
                        path_weight: SUM(FOR edge IN p.edges RETURN edge.weight)
                    }}
        )
        FOR r IN results
            LIMIT @limit
            RETURN r
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={
                'start_nodes': start_docs,
                'max_depth': min(max_depth, 2),  # Cap at 2 hops for safety
                'limit': limit
            },
            ttl=30  # 30 second timeout
        )
        
        return list(cursor)
    
    def personalized_pagerank(
        self,
        seed_nodes: List[str],
        alpha: float = 0.15,
        max_iterations: int = 20,
        tolerance: float = 1e-6
    ) -> Dict[str, float]:
        """
        Compute personalized PageRank from seed nodes
        Returns dict of node_id -> score
        """
        # This is a simplified implementation
        # For production, use Pregel or native graph algorithms
        
        query = f"""
        LET seeds = @seed_nodes
        LET graph_nodes = (
            FOR doc IN {self.config.nodes_collection}
                RETURN doc._key
        )
        
        LET teleport = {{}}
        LET init_scores = (
            FOR node IN graph_nodes
                RETURN {{
                    [node]: (node IN seeds ? 1.0 / LENGTH(seeds) : 0.0)
                }}
        )
        
        RETURN MERGE(init_scores)
        """
        
        # Simplified - return seed nodes with equal weight
        # Full implementation would iterate
        result = {}
        for node in seed_nodes:
            result[node] = 1.0 / len(seed_nodes)
        
        return result
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Retrieve a node by ID"""
        try:
            return self.nodes_collection.get(node_id)
        except:
            return None
    
    def get_neighbors(self, node_id: str, direction: str = "any") -> List[Dict]:
        """Get neighbors of a node"""
        node_doc = f"{self.config.nodes_collection}/{node_id}"
        
        direction_clause = {
            "outbound": "OUTBOUND",
            "inbound": "INBOUND",
            "any": "ANY"
        }[direction.lower()]
        
        query = f"""
        FOR v, e IN 1..1 {direction_clause} @start {self.config.edges_collection}
            RETURN {{
                node: v,
                edge: e
            }}
        """
        
        cursor = self.db.aql.execute(
            query,
            bind_vars={'start': node_doc}
        )
        
        return list(cursor)
    
    def update_edge_weight(self, source_id: str, target_id: str, new_weight: float):
        """Update edge weight (for dynamic learning)"""
        query = f"""
        FOR edge IN {self.config.edges_collection}
            FILTER edge._from == @from AND edge._to == @to
            UPDATE edge WITH {{ weight: @weight }} IN {self.config.edges_collection}
        """
        
        self.db.aql.execute(
            query,
            bind_vars={
                'from': f"{self.config.nodes_collection}/{source_id}",
                'to': f"{self.config.nodes_collection}/{target_id}",
                'weight': new_weight
            }
        )
    
    def fulltext_search(self, query_text: str, top_k: int = 10) -> List[Dict]:
        """Full-text search on content"""
        aql_query = f"""
        FOR doc IN FULLTEXT({self.config.nodes_collection}, 'content', @query)
            LIMIT @top_k
            RETURN {{
                node_id: doc._key,
                content: doc.content,
                modality: doc.modality
            }}
        """
        
        cursor = self.db.aql.execute(
            aql_query,
            bind_vars={'query': query_text, 'top_k': top_k}
        )
        
        return list(cursor)
    
    def get_statistics(self) -> Dict:
        """Get database statistics"""
        return {
            'total_nodes': self.nodes_collection.count(),
            'total_edges': self.edges_collection.count(),
            'modality_distribution': self._get_modality_distribution(),
            'resolution_distribution': self._get_resolution_distribution()
        }
    
    def _get_modality_distribution(self) -> Dict[str, int]:
        """Get distribution of nodes by modality"""
        query = f"""
        FOR doc IN {self.config.nodes_collection}
            COLLECT modality = doc.modality WITH COUNT INTO count
            RETURN {{ modality: modality, count: count }}
        """
        
        cursor = self.db.aql.execute(query)
        return {r['modality']: r['count'] for r in cursor}
    
    def _get_resolution_distribution(self) -> Dict[str, int]:
        """Get distribution of nodes by resolution"""
        query = f"""
        FOR doc IN {self.config.nodes_collection}
            COLLECT resolution = doc.resolution WITH COUNT INTO count
            RETURN {{ resolution: resolution, count: count }}
        """
        
        cursor = self.db.aql.execute(query)
        return {r['resolution']: r['count'] for r in cursor}
    
    def list_sources(self) -> List[str]:
        """List all unique sources in the database"""
        query = f"""
        FOR doc IN {self.config.nodes_collection}
            RETURN DISTINCT doc.source
        """
        cursor = self.db.aql.execute(query)
        return [s for s in cursor if s]

    def delete_document(self, source_name: str) -> int:
        """
        Delete all nodes and edges associated with a source
        Returns number of deleted nodes
        """
        # 1. Get all node IDs for this source
        id_query = f"FOR doc IN {self.config.nodes_collection} FILTER doc.source == @source RETURN doc._id"
        cursor = self.db.aql.execute(id_query, bind_vars={'source': source_name})
        node_ids = list(cursor)
        
        if not node_ids:
            return 0
            
        # 2. Delete edges connected to these nodes
        # Use a single query to find and remove all relevant edges
        edge_query = f"""
        FOR e IN {self.config.edges_collection}
            FILTER e._from IN @node_ids OR e._to IN @node_ids
            REMOVE e IN {self.config.edges_collection}
            OPTIONS {{ ignoreErrors: true }}
        """
        self.db.aql.execute(edge_query, bind_vars={'node_ids': node_ids})
        
        # 3. Delete the nodes
        node_removal_query = f"""
        FOR doc IN {self.config.nodes_collection}
            FILTER doc._id IN @node_ids
            REMOVE doc IN {self.config.nodes_collection}
            OPTIONS {{ ignoreErrors: true }}
        """
        self.db.aql.execute(node_removal_query, bind_vars={'node_ids': node_ids})
        
        logger.info(f"Deleted source '{source_name}': {len(node_ids)} nodes removed.")
        return len(node_ids)

    def clear_database(self):
        """Clear all data from database"""
        self.db.collection(self.config.nodes_collection).truncate()
        self.db.collection(self.config.edges_collection).truncate()
        logger.info("Database cleared")

    def save_api_key(self, provider: str, name: str, key: str, model_name: Optional[str] = None):
        """Persistently save an API key"""
        coll = self.db.collection("system_config")
        # Sanitize name for use in _key (ArangoDB keys can't have spaces)
        safe_name = "".join([c if c.isalnum() or c in "-_" else "_" for c in name])
        key_id = f"api_key_{provider.lower()}_{safe_name}"
        data = {
            "_key": key_id,
            "type": "api_key",
            "provider": provider,
            "name": name, # Keep original name in data
            "key": key,
            "model_name": model_name
        }
        coll.insert(data, overwrite=True)

    def list_api_keys(self) -> List[Dict]:
        """List all persistent API keys"""
        query = "FOR doc IN system_config FILTER doc.type == 'api_key' RETURN doc"
        cursor = self.db.aql.execute(query)
        return [doc for doc in cursor]

    def delete_api_key(self, provider: str, name: str):
        """Delete a persistent API key"""
        # Use same sanitization logic
        safe_name = "".join([c if c.isalnum() or c in "-_" else "_" for c in name])
        key_id = f"api_key_{provider.lower()}_{safe_name}"
        coll = self.db.collection("system_config")
        if coll.has(key_id):
            coll.delete(key_id)

    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("Closed ArangoDB connection")