"""
Analysis service layer with dependency injection
"""

import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from core.interfaces import IAnalysisRepository, IImageProcessor, CVAnalysisEntity

logger = logging.getLogger(__name__)


class AnalysisService:
    """Service layer for CV analysis operations"""
    
    def __init__(self, repository: IAnalysisRepository, image_processor: IImageProcessor):
        """Initialize with injected dependencies"""
        self.repository = repository
        self.image_processor = image_processor
        logger.info(f"AnalysisService initialized with {type(repository).__name__} and {type(image_processor).__name__}")
    
    async def analyze_and_save_image(self, image_data: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze image and save results to repository"""
        try:
            # Process the image
            start_time = datetime.now()
            analysis_results = await self.image_processor.analyze_image(image_data)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            # Calculate image metadata
            image_metadata = self._calculate_image_metadata(image_data, metadata or {})
            
            # Add processing information
            processing_info = {
                'processing_time_ms': processing_time,
                'model_version': analysis_results.get('model_info', {}).get('version', 'Smart CV v1.0'),
                'overall_confidence': analysis_results.get('processing_info', {}).get('overall_confidence', 0.8),
                'timestamp': datetime.now().isoformat()
            }
            
            # Create entity
            entity = CVAnalysisEntity(
                timestamp=datetime.now(),
                image_metadata=image_metadata,
                processing_info=processing_info,
                detection_summary=analysis_results.get('detection_summary', {}),
                people_analysis=analysis_results.get('people', []),
                scene_analysis=analysis_results.get('scene_analysis', {})
            )
            
            # Save to repository
            analysis_id = await self.repository.save_analysis(entity)
            
            # Return response with ID
            response = {
                'success': True,
                'analysis_id': analysis_id,
                'processing_time_ms': processing_time,
                **analysis_results
            }
            
            logger.info(f"Image analyzed and saved with ID: {analysis_id}")
            return response
            
        except Exception as e:
            logger.error(f"Error in analyze_and_save_image: {e}")
            raise
    
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by ID"""
        try:
            entity = await self.repository.get_analysis_by_id(analysis_id)
            if not entity:
                return None
            
            return self._entity_to_response(entity)
            
        except Exception as e:
            logger.error(f"Error getting analysis {analysis_id}: {e}")
            raise
    
    async def get_all_analyses(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all analyses"""
        try:
            entities = await self.repository.get_all_analyses(limit)
            return [self._entity_to_response(entity) for entity in entities]
            
        except Exception as e:
            logger.error(f"Error getting all analyses: {e}")
            raise
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis by ID"""
        try:
            return await self.repository.delete_analysis(analysis_id)
            
        except Exception as e:
            logger.error(f"Error deleting analysis {analysis_id}: {e}")
            raise
    
    async def search_analyses(self, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search analyses with filters"""
        try:
            entities = await self.repository.get_analyses_by_filter(filters)
            return [self._entity_to_response(entity) for entity in entities]
            
        except Exception as e:
            logger.error(f"Error searching analyses: {e}")
            raise
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        try:
            all_analyses = await self.repository.get_all_analyses()
            
            if not all_analyses:
                return {
                    'total_analyses': 0,
                    'total_people_detected': 0,
                    'average_confidence': 0,
                    'average_processing_time': 0
                }
            
            total_people = sum(
                entity.detection_summary.get('people_detected', 0) 
                for entity in all_analyses
            )
            
            confidences = [
                entity.detection_summary.get('average_confidence', 0)
                for entity in all_analyses
                if entity.detection_summary
            ]
            
            processing_times = [
                entity.processing_info.get('processing_time_ms', 0)
                for entity in all_analyses
                if entity.processing_info
            ]
            
            return {
                'total_analyses': len(all_analyses),
                'total_people_detected': total_people,
                'average_confidence': sum(confidences) / len(confidences) if confidences else 0,
                'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get system capabilities"""
        return self.image_processor.get_capabilities()
    
    def _calculate_image_metadata(self, image_data: str, additional_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate image metadata from base64 data"""
        try:
            # Estimate size from base64 (rough calculation)
            image_size = len(image_data) * 3 // 4 if image_data else 0
            
            metadata = {
                'size': image_size,
                'type': 'image/jpeg',  # Default
                'format': 'base64',
                **additional_metadata
            }
            
            return metadata
            
        except Exception as e:
            logger.warning(f"Error calculating image metadata: {e}")
            return additional_metadata
    
    def _entity_to_response(self, entity: CVAnalysisEntity) -> Dict[str, Any]:
        """Convert entity to API response format"""
        return {
            'id': entity.id,
            'timestamp': entity.timestamp.isoformat() if entity.timestamp else None,
            'image_metadata': entity.image_metadata,
            'processing_info': entity.processing_info,
            'detection_summary': entity.detection_summary,
            'people': entity.people_analysis,
            'scene_analysis': entity.scene_analysis
        } 