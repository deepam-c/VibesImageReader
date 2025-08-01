"""
MongoDB implementation of the analysis repository
"""

import asyncio
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.interfaces import IAnalysisRepository, CVAnalysisEntity

logger = logging.getLogger(__name__)


class MongoDBAnalysisRepository(IAnalysisRepository):
    """MongoDB implementation of analysis repository"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.client = None
        self.db = None
        self.collection = None
        self._initialize_mongodb()
    
    def _initialize_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            from pymongo import MongoClient
            from bson import ObjectId
            
            # Connection string
            connection_string = self.config.get(
                'connection_string', 
                'mongodb://localhost:27017/'
            )
            
            # Connect to MongoDB
            self.client = MongoClient(connection_string)
            
            # Get database and collection
            database_name = self.config.get('database_name', 'cv_analytics')
            collection_name = self.config.get('collection_name', 'analyses')
            
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            
            # Create indexes for better performance
            self.collection.create_index([("timestamp", -1)])
            self.collection.create_index([("processing_info.model_version", 1)])
            
            logger.info(f"MongoDB initialized: {database_name}.{collection_name}")
            
        except ImportError:
            logger.error("PyMongo not installed. Install with: pip install pymongo")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    async def save_analysis(self, analysis: CVAnalysisEntity) -> str:
        """Save analysis to MongoDB"""
        try:
            from bson import ObjectId
            
            # Convert entity to dictionary
            data = self._entity_to_dict(analysis)
            data['timestamp'] = analysis.timestamp or datetime.now()
            
            # Insert document
            result = self.collection.insert_one(data)
            analysis_id = str(result.inserted_id)
            
            logger.info(f"Analysis saved to MongoDB with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error saving analysis to MongoDB: {e}")
            raise
    
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[CVAnalysisEntity]:
        """Get analysis by ID from MongoDB"""
        try:
            from bson import ObjectId
            
            # Convert string ID to ObjectId
            try:
                object_id = ObjectId(analysis_id)
            except Exception:
                # If it's not a valid ObjectId, search by string ID
                document = self.collection.find_one({"_id": analysis_id})
            else:
                document = self.collection.find_one({"_id": object_id})
            
            if not document:
                return None
            
            return self._document_to_entity(document)
            
        except Exception as e:
            logger.error(f"Error getting analysis from MongoDB: {e}")
            raise
    
    async def get_all_analyses(self, limit: Optional[int] = None) -> List[CVAnalysisEntity]:
        """Get all analyses from MongoDB"""
        try:
            # Create query with sorting
            cursor = self.collection.find().sort("timestamp", -1)
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert documents to entities
            analyses = []
            for document in cursor:
                analyses.append(self._document_to_entity(document))
            
            logger.info(f"Retrieved {len(analyses)} analyses from MongoDB")
            return analyses
            
        except Exception as e:
            logger.error(f"Error getting all analyses from MongoDB: {e}")
            raise
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis from MongoDB"""
        try:
            from bson import ObjectId
            
            # Try to convert to ObjectId first
            try:
                object_id = ObjectId(analysis_id)
                result = self.collection.delete_one({"_id": object_id})
            except Exception:
                # If not a valid ObjectId, try string ID
                result = self.collection.delete_one({"_id": analysis_id})
            
            success = result.deleted_count > 0
            
            if success:
                logger.info(f"Analysis {analysis_id} deleted from MongoDB")
            else:
                logger.warning(f"Analysis {analysis_id} not found for deletion")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting analysis from MongoDB: {e}")
            return False
    
    async def get_analyses_by_filter(self, filters: Dict[str, Any]) -> List[CVAnalysisEntity]:
        """Get analyses by custom filters"""
        try:
            # MongoDB supports rich querying
            cursor = self.collection.find(filters).sort("timestamp", -1)
            
            analyses = []
            for document in cursor:
                analyses.append(self._document_to_entity(document))
            
            return analyses
            
        except Exception as e:
            logger.error(f"Error filtering analyses in MongoDB: {e}")
            raise
    
    def _entity_to_dict(self, entity: CVAnalysisEntity) -> Dict[str, Any]:
        """Convert entity to dictionary"""
        data = {
            'image_metadata': entity.image_metadata or {},
            'processing_info': entity.processing_info or {},
            'detection_summary': entity.detection_summary or {},
            'people_analysis': entity.people_analysis or [],
            'scene_analysis': entity.scene_analysis or {}
        }
        
        # Add ID if provided
        if entity.id:
            data['_id'] = entity.id
        
        return data
    
    def _document_to_entity(self, document: Dict[str, Any]) -> CVAnalysisEntity:
        """Convert MongoDB document to entity"""
        return CVAnalysisEntity(
            id=str(document.get('_id')),
            timestamp=document.get('timestamp'),
            image_metadata=document.get('image_metadata'),
            processing_info=document.get('processing_info'),
            detection_summary=document.get('detection_summary'),
            people_analysis=document.get('people_analysis'),
            scene_analysis=document.get('scene_analysis')
        ) 