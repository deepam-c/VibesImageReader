"""
Firebase implementation of the analysis repository
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.interfaces import IAnalysisRepository, CVAnalysisEntity

logger = logging.getLogger(__name__)


class FirebaseAnalysisRepository(IAnalysisRepository):
    """Firebase implementation of analysis repository"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._db = None
        self._initialize_firebase()
    
    def _initialize_firebase(self):
        """Initialize Firebase connection"""
        try:
            from firebase_admin import credentials, firestore, initialize_app
            import firebase_admin
            
            # Check if Firebase is already initialized
            try:
                firebase_admin.get_app()
                logger.info("Firebase already initialized")
            except ValueError:
                # Initialize Firebase
                if 'service_account_path' in self.config:
                    cred = credentials.Certificate(self.config['service_account_path'])
                    initialize_app(cred)
                else:
                    # Use default credentials or config
                    initialize_app()
                logger.info("Firebase initialized successfully")
            
            self._db = firestore.client()
            logger.info("Firestore client initialized")
            
        except ImportError:
            logger.error("Firebase Admin SDK not installed. Install with: pip install firebase-admin")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize Firebase: {e}")
            raise
    
    async def save_analysis(self, analysis: CVAnalysisEntity) -> str:
        """Save analysis to Firestore"""
        try:
            # Convert entity to dictionary
            data = self._entity_to_dict(analysis)
            data['timestamp'] = datetime.now()
            
            # Remove None values
            data = self._clean_none_values(data)
            
            # Add to Firestore
            doc_ref = self._db.collection('cvAnalyses').add(data)
            doc_id = doc_ref[1].id
            
            logger.info(f"Analysis saved to Firebase with ID: {doc_id}")
            return doc_id
            
        except Exception as e:
            logger.error(f"Error saving analysis to Firebase: {e}")
            raise
    
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[CVAnalysisEntity]:
        """Get analysis by ID from Firestore"""
        try:
            doc_ref = self._db.collection('cvAnalyses').document(analysis_id)
            doc = doc_ref.get()
            
            if not doc.exists:
                return None
            
            data = doc.to_dict()
            data['id'] = doc.id
            
            return self._dict_to_entity(data)
            
        except Exception as e:
            logger.error(f"Error getting analysis from Firebase: {e}")
            raise
    
    async def get_all_analyses(self, limit: Optional[int] = None) -> List[CVAnalysisEntity]:
        """Get all analyses from Firestore"""
        try:
            query = self._db.collection('cvAnalyses').order_by('timestamp', direction=firestore.Query.DESCENDING)
            
            if limit:
                query = query.limit(limit)
            
            docs = query.stream()
            
            analyses = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                analyses.append(self._dict_to_entity(data))
            
            logger.info(f"Retrieved {len(analyses)} analyses from Firebase")
            return analyses
            
        except Exception as e:
            logger.error(f"Error getting all analyses from Firebase: {e}")
            raise
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis from Firestore"""
        try:
            doc_ref = self._db.collection('cvAnalyses').document(analysis_id)
            doc_ref.delete()
            
            logger.info(f"Analysis {analysis_id} deleted from Firebase")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting analysis from Firebase: {e}")
            return False
    
    async def get_analyses_by_filter(self, filters: Dict[str, Any]) -> List[CVAnalysisEntity]:
        """Get analyses by custom filters"""
        try:
            query = self._db.collection('cvAnalyses')
            
            # Apply filters
            for field, value in filters.items():
                query = query.where(field, '==', value)
            
            docs = query.stream()
            
            analyses = []
            for doc in docs:
                data = doc.to_dict()
                data['id'] = doc.id
                analyses.append(self._dict_to_entity(data))
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error filtering analyses in Firebase: {e}")
            raise
    
    def _entity_to_dict(self, entity: CVAnalysisEntity) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        return {
            'image_metadata': entity.image_metadata or {},
            'processing_info': entity.processing_info or {},
            'detection_summary': entity.detection_summary or {},
            'people_analysis': entity.people_analysis or [],
            'scene_analysis': entity.scene_analysis or {}
        }
    
    def _dict_to_entity(self, data: Dict[str, Any]) -> CVAnalysisEntity:
        """Convert dictionary to entity"""
        return CVAnalysisEntity(
            id=data.get('id'),
            timestamp=data.get('timestamp'),
            image_metadata=data.get('image_metadata'),
            processing_info=data.get('processing_info'),
            detection_summary=data.get('detection_summary'),
            people_analysis=data.get('people_analysis'),
            scene_analysis=data.get('scene_analysis')
        )
    
    def _clean_none_values(self, obj: Any) -> Any:
        """Remove None values from nested objects"""
        if isinstance(obj, dict):
            return {k: self._clean_none_values(v) for k, v in obj.items() if v is not None}
        elif isinstance(obj, list):
            return [self._clean_none_values(item) for item in obj if item is not None]
        return obj 