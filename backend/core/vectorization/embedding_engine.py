import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
from typing import Dict, List, Optional
from loguru import logger
import hashlib
from pathlib import Path

from backend.models.config import Modality


class VectorizationEngine:
    """
    Multimodal embedding engine
    Generates embeddings for text, images, audio, video, tables
    Uses different models for different modalities
    """
    
    def __init__(self, cache_dir: str = "./data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Vectorization engine using device: {self.device}")
        
        # Model storage
        self.models: Dict[str, any] = {}
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Load all embedding models"""
        try:
            # Text embeddings - E5-large
            logger.info("Loading text embedding model (E5-large)...")
            self.models['text'] = SentenceTransformer(
                'intfloat/e5-large-v2',
                cache_folder=str(self.cache_dir),
                device=self.device
            )
            
            # Image embeddings - CLIP
            logger.info("Loading image embedding model (CLIP)...")
            self.models['clip_processor'] = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=str(self.cache_dir)
            )
            self.models['clip_model'] = CLIPModel.from_pretrained(
                "openai/clip-vit-base-patch32",
                cache_dir=str(self.cache_dir)
            ).to(self.device)
            
            # Audio embeddings - Use Whisper embeddings
            # Video embeddings - Use CLIP on keyframes
            # Table embeddings - Use text model on serialized tables
            
            logger.info("All embedding models loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise
    
    def embed_text(self, texts: List[str], prefix: str = "passage: ") -> np.ndarray:
        """
        Generate text embeddings using E5-large
        E5 requires prefixes for optimal performance
        
        Args:
            texts: List of text strings
            prefix: Prefix for E5 model (default: "passage: ")
        
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        if not texts:
            return np.array([])
        
        # Add E5 prefix
        prefixed_texts = [prefix + text for text in texts]
        
        # Generate embeddings
        embeddings = self.models['text'].encode(
            prefixed_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Embed a query string
        Uses "query: " prefix for E5
        """
        return self.embed_text([query], prefix="query: ")[0]
    
    def embed_image(self, image) -> np.ndarray:
        """
        Generate image embedding using CLIP
        
        Args:
            image: PIL Image or numpy array
        
        Returns:
            numpy array of shape (embedding_dim,)
        """
        # Process image
        inputs = self.models['clip_processor'](
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate embedding
        with torch.no_grad():
            image_features = self.models['clip_model'].get_image_features(**inputs)
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def embed_image_text_aligned(self, image, text: str) -> Dict[str, np.ndarray]:
        """
        Generate aligned embeddings for image and text using CLIP
        Useful for cross-modal retrieval
        
        Returns:
            Dict with 'image' and 'text' embeddings in same space
        """
        inputs = self.models['clip_processor'](
            text=[text],
            images=image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.models['clip_model'](**inputs)
            
            image_embeds = outputs.image_embeds
            text_embeds = outputs.text_embeds
            
            # Normalize
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        
        return {
            'image': image_embeds.cpu().numpy()[0],
            'text': text_embeds.cpu().numpy()[0]
        }
    
    def embed_audio_transcript(self, transcript: str) -> np.ndarray:
        """
        Embed audio via its transcript
        Uses text model
        """
        return self.embed_text([transcript], prefix="passage: ")[0]
    
    def embed_video_keyframes(self, frames: List) -> np.ndarray:
        """
        Embed video by averaging keyframe embeddings
        
        Args:
            frames: List of PIL Images (keyframes)
        
        Returns:
            Averaged embedding
        """
        if not frames:
            return np.zeros(512)  # CLIP embedding size
        
        frame_embeddings = [self.embed_image(frame) for frame in frames]
        
        # Average pooling
        avg_embedding = np.mean(frame_embeddings, axis=0)
        
        # Re-normalize
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        return avg_embedding
    
    def embed_table(self, table_data: str) -> np.ndarray:
        """
        Embed tabular data
        Serializes table and uses text embedding
        
        Args:
            table_data: Serialized table (e.g., CSV format or schema description)
        """
        return self.embed_text([table_data], prefix="passage: ")[0]
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings
        """
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def batch_compute_similarities(
        self, 
        query_emb: np.ndarray, 
        candidate_embs: np.ndarray
    ) -> np.ndarray:
        """
        Compute similarities between query and multiple candidates
        
        Args:
            query_emb: (embedding_dim,)
            candidate_embs: (n_candidates, embedding_dim)
        
        Returns:
            Array of similarities (n_candidates,)
        """
        # Normalize
        query_norm = query_emb / np.linalg.norm(query_emb)
        candidate_norms = candidate_embs / np.linalg.norm(candidate_embs, axis=1, keepdims=True)
        
        # Compute dot products
        similarities = np.dot(candidate_norms, query_norm)
        
        return similarities
    
    def get_embedding_dimension(self, modality: Modality) -> int:
        """Get embedding dimension for a modality"""
        dim_map = {
            Modality.TEXT: 1024,  # E5-large
            Modality.IMAGE: 512,  # CLIP
            Modality.AUDIO: 1024,  # Text model (transcript)
            Modality.VIDEO: 512,  # CLIP (keyframes)
            Modality.TABLE: 1024,  # Text model
        }
        return dim_map.get(modality, 1024)
    
    def create_multi_vector_embedding(
        self, 
        content_dict: Dict[str, any]
    ) -> Dict[str, np.ndarray]:
        """
        Create multiple embeddings for multimodal content
        
        Args:
            content_dict: Dict with keys like 'text', 'image', 'audio_transcript'
        
        Returns:
            Dict of embedding_type -> embedding
        """
        embeddings = {}
        
        if 'text' in content_dict and content_dict['text']:
            embeddings['text'] = self.embed_text([content_dict['text']])[0]
        
        if 'image' in content_dict and content_dict['image'] is not None:
            embeddings['image'] = self.embed_image(content_dict['image'])
        
        if 'audio_transcript' in content_dict and content_dict['audio_transcript']:
            embeddings['audio'] = self.embed_audio_transcript(
                content_dict['audio_transcript']
            )
        
        if 'table_data' in content_dict and content_dict['table_data']:
            embeddings['table'] = self.embed_table(content_dict['table_data'])
        
        return embeddings
    
    def cache_embedding(self, content_hash: str, embedding: np.ndarray):
        """Cache embedding to disk"""
        cache_path = self.cache_dir / f"{content_hash}.npy"
        np.save(cache_path, embedding)
    
    def load_cached_embedding(self, content_hash: str) -> Optional[np.ndarray]:
        """Load cached embedding"""
        cache_path = self.cache_dir / f"{content_hash}.npy"
        if cache_path.exists():
            return np.load(cache_path)
        return None
    
    @staticmethod
    def hash_content(content: str) -> str:
        """Generate hash for content"""
        return hashlib.md5(content.encode()).hexdigest()