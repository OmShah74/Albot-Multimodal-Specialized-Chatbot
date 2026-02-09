import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any
import time
from PIL import Image
import whisper
import cv2
from moviepy.editor import VideoFileClip
import pandas as pd
from pypdf import PdfReader
import pdfplumber
from docx import Document
from pptx import Presentation
import yt_dlp
import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright
from loguru import logger

from backend.models.config import KnowledgeAtom, Modality, Resolution


class MultimodalProcessor:
    """Universal ingestion for all modalities"""
    
    def __init__(self, data_dir: str = "./data/uploads"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load Whisper for audio/video transcription
        logger.info("Loading Whisper model...")
        self.whisper_model = whisper.load_model("base")
        
    def process_file(self, file_path: str, source_name: str = None) -> List[KnowledgeAtom]:
        """Route file to appropriate processor"""
        path = Path(file_path)
        source = source_name or path.name
        
        ext = path.suffix.lower()
        
        if ext in ['.txt', '.md']:
            return self.process_text_file(path, source)
        elif ext == '.pdf':
            return self.process_pdf(path, source)
        elif ext in ['.doc', '.docx']:
            return self.process_docx(path, source)
        elif ext in ['.ppt', '.pptx']:
            return self.process_pptx(path, source)
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp']:
            return self.process_image(path, source)
        elif ext in ['.mp3', '.wav', '.m4a', '.ogg']:
            return self.process_audio(path, source)
        elif ext in ['.mp4', '.avi', '.mov', '.mkv']:
            return self.process_video(path, source)
        elif ext in ['.csv', '.tsv']:
            return self.process_csv(path, source)
        elif ext in ['.xlsx', '.xls']:
            return self.process_excel(path, source)
        else:
            logger.warning(f"Unsupported file type: {ext}")
            return []
    
    def process_text_file(self, path: Path, source: str) -> List[KnowledgeAtom]:
        """Process plain text files"""
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        atoms = []
        timestamp = time.time()
        
        # Fine: sentences
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        for i, sent in enumerate(sentences):
            if len(sent) > 10:
                atoms.append(KnowledgeAtom(
                    content=sent,
                    modality=Modality.TEXT,
                    resolution=Resolution.FINE,
                    source=source,
                    timestamp=timestamp + i,
                    metadata={'sentence_idx': i}
                ))
        
        # Mid: paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs):
            if len(para) > 50:
                atoms.append(KnowledgeAtom(
                    content=para,
                    modality=Modality.TEXT,
                    resolution=Resolution.MID,
                    source=source,
                    timestamp=timestamp + i * 10,
                    metadata={'paragraph_idx': i}
                ))
        
        # Coarse: full document
        atoms.append(KnowledgeAtom(
            content=content[:5000],  # Truncate if too long
            modality=Modality.TEXT,
            resolution=Resolution.COARSE,
            source=source,
            timestamp=timestamp,
            metadata={'full_document': True}
        ))
        
        return atoms
    
    def process_pdf(self, path: Path, source: str) -> List[KnowledgeAtom]:
        """
        Process PDF with granular multi-resolution extraction.
        
        Creates:
        - FINE: Individual sentences (>20 chars)
        - MID: Paragraphs and page sections
        - COARSE: Full document summary
        """
        atoms = []
        timestamp = time.time()
        
        try:
            with pdfplumber.open(path) as pdf:
                full_text = ""
                sentence_idx = 0
                paragraph_idx = 0
                
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text() or ""
                    full_text += page_text + "\n\n"
                    
                    if not page_text.strip():
                        continue
                    
                    # --- FINE RESOLUTION: Sentences ---
                    # Split by sentence-ending punctuation
                    sentences = []
                    for sent in page_text.replace('\n', ' ').split('.'):
                        sent = sent.strip()
                        if len(sent) > 20:  # Filter very short fragments
                            sentences.append(sent + '.')
                    
                    for sent in sentences:
                        atoms.append(KnowledgeAtom(
                            content=sent,
                            modality=Modality.TEXT,
                            resolution=Resolution.FINE,
                            source=source,
                            timestamp=timestamp + sentence_idx * 0.01,
                            metadata={
                                'page_num': page_num + 1,
                                'sentence_idx': sentence_idx,
                                'type': 'sentence'
                            }
                        ))
                        sentence_idx += 1
                    
                    # --- MID RESOLUTION: Paragraphs ---
                    # Split by double newlines or significant whitespace
                    paragraphs = [p.strip() for p in page_text.split('\n\n') if p.strip()]
                    
                    # If no double-newline paragraphs, try single newlines
                    if len(paragraphs) <= 1:
                        paragraphs = [p.strip() for p in page_text.split('\n') if len(p.strip()) > 50]
                    
                    for para in paragraphs:
                        if len(para) > 50:  # Minimum paragraph length
                            atoms.append(KnowledgeAtom(
                                content=para,
                                modality=Modality.TEXT,
                                resolution=Resolution.MID,
                                source=source,
                                timestamp=timestamp + page_num + paragraph_idx * 0.1,
                                metadata={
                                    'page_num': page_num + 1,
                                    'paragraph_idx': paragraph_idx,
                                    'type': 'paragraph'
                                }
                            ))
                            paragraph_idx += 1
                    
                    # --- MID RESOLUTION: Full Page ---
                    atoms.append(KnowledgeAtom(
                        content=page_text,
                        modality=Modality.TEXT,
                        resolution=Resolution.MID,
                        source=source,
                        timestamp=timestamp + page_num,
                        metadata={
                            'page_num': page_num + 1,
                            'type': 'page'
                        }
                    ))
                
                # --- COARSE RESOLUTION: Document Summary ---
                # Store multiple chunks if document is long
                doc_chunks = [full_text[i:i+4000] for i in range(0, min(len(full_text), 20000), 4000)]
                
                for chunk_idx, chunk in enumerate(doc_chunks):
                    atoms.append(KnowledgeAtom(
                        content=chunk,
                        modality=Modality.TEXT,
                        resolution=Resolution.COARSE,
                        source=source,
                        timestamp=timestamp + chunk_idx * 0.001,
                        metadata={
                            'total_pages': len(pdf.pages),
                            'chunk_idx': chunk_idx,
                            'type': 'document_chunk'
                        }
                    ))
                
                logger.info(f"PDF '{source}': {sentence_idx} sentences, {paragraph_idx} paragraphs, {len(pdf.pages)} pages, {len(doc_chunks)} doc chunks")
                
        except Exception as e:
            logger.error(f"PDF processing error: {e}")
        
        return atoms
    
    def process_image(self, path: Path, source: str) -> List[KnowledgeAtom]:
        """Process images"""
        atoms = []
        timestamp = time.time()
        
        # Load image
        image = Image.open(path)
        
        # Create atom with image metadata
        atoms.append(KnowledgeAtom(
            content=f"Image: {path.name}",
            modality=Modality.IMAGE,
            resolution=Resolution.FINE,
            source=source,
            timestamp=timestamp,
            metadata={
                'path': str(path),
                'size': image.size,
                'format': image.format
            }
        ))
        
        return atoms
    
    def process_audio(self, path: Path, source: str) -> List[KnowledgeAtom]:
        """Process audio with transcription"""
        atoms = []
        timestamp = time.time()
        
        try:
            # Transcribe
            result = self.whisper_model.transcribe(str(path))
            transcript = result['text']
            segments = result.get('segments', [])
            
            # Segment-level (FINE)
            for seg in segments:
                atoms.append(KnowledgeAtom(
                    content=seg['text'],
                    modality=Modality.AUDIO,
                    resolution=Resolution.FINE,
                    source=source,
                    timestamp=timestamp + seg['start'],
                    metadata={
                        'start': seg['start'],
                        'end': seg['end'],
                        'duration': seg['end'] - seg['start']
                    }
                ))
            
            # Full transcript (COARSE)
            atoms.append(KnowledgeAtom(
                content=transcript,
                modality=Modality.AUDIO,
                resolution=Resolution.COARSE,
                source=source,
                timestamp=timestamp,
                metadata={'full_transcript': True}
            ))
            
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
        
        return atoms
    
    def process_video(self, path: Path, source: str) -> List[KnowledgeAtom]:
        """Process video: keyframes + transcription"""
        atoms = []
        timestamp = time.time()
        
        try:
            # Extract audio and transcribe
            clip = VideoFileClip(str(path))
            audio_path = path.with_suffix('.wav')
            clip.audio.write_audiofile(str(audio_path), verbose=False, logger=None)
            
            # Transcribe
            result = self.whisper_model.transcribe(str(audio_path))
            transcript = result['text']
            
            # Transcript atom
            atoms.append(KnowledgeAtom(
                content=transcript,
                modality=Modality.TEXT,
                resolution=Resolution.COARSE,
                source=source,
                timestamp=timestamp,
                metadata={'video_transcript': True}
            ))
            
            # Extract keyframes every 5 seconds
            duration = clip.duration
            for t in np.arange(0, duration, 5.0):
                frame = clip.get_frame(t)
                atoms.append(KnowledgeAtom(
                    content=f"Video frame at {t}s",
                    modality=Modality.VIDEO,
                    resolution=Resolution.FINE,
                    source=source,
                    timestamp=timestamp + t,
                    metadata={'frame_time': t, 'frame_data': frame}
                ))
            
            clip.close()
            audio_path.unlink()
            
        except Exception as e:
            logger.error(f"Video processing error: {e}")
        
        return atoms
    
    def process_csv(self, path: Path, source: str) -> List[KnowledgeAtom]:
        """Process CSV/tabular data"""
        atoms = []
        timestamp = time.time()
        
        df = pd.read_csv(path)
        
        # Schema atom
        schema = f"Columns: {', '.join(df.columns)}\nShape: {df.shape}"
        atoms.append(KnowledgeAtom(
            content=schema,
            modality=Modality.TABLE,
            resolution=Resolution.COARSE,
            source=source,
            timestamp=timestamp,
            metadata={'schema': True, 'rows': len(df), 'cols': len(df.columns)}
        ))
        
        # Row-level atoms (sample)
        for idx, row in df.head(100).iterrows():
            row_text = ', '.join([f"{col}: {val}" for col, val in row.items()])
            atoms.append(KnowledgeAtom(
                content=row_text,
                modality=Modality.TABLE,
                resolution=Resolution.FINE,
                source=source,
                timestamp=timestamp + idx,
                metadata={'row_idx': idx}
            ))
        
        return atoms
    
    def process_url(self, url: str) -> List[KnowledgeAtom]:
        """Process web URL"""
        atoms = []
        timestamp = time.time()
        
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=30000)
                
                content = page.content()
                browser.close()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(content, 'html.parser')
            
            # Extract text
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = '\n'.join(lines)
            
            # Create atoms
            atoms.append(KnowledgeAtom(
                content=text[:5000],
                modality=Modality.TEXT,
                resolution=Resolution.COARSE,
                source=url,
                timestamp=timestamp,
                metadata={'url': url}
            ))
            
        except Exception as e:
            logger.error(f"URL processing error: {e}")
        
        return atoms
    
    def process_youtube(self, url: str) -> List[KnowledgeAtom]:
        """Process YouTube video"""
        atoms = []
        timestamp = time.time()
        
        try:
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'mp3',
                }],
                'outtmpl': str(self.data_dir / '%(id)s.%(ext)s'),
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                audio_file = self.data_dir / f"{info['id']}.mp3"
            
            # Process as audio
            atoms = self.process_audio(audio_file, url)
            
            # Clean up
            if audio_file.exists():
                audio_file.unlink()
            
        except Exception as e:
            logger.error(f"YouTube processing error: {e}")
        
        return atoms