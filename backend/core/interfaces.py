"""
Core interfaces for dependency injection and repository pattern
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass


@dataclass
class CVAnalysisEntity:
    """Domain entity for CV Analysis data"""
    id: Optional[str] = None
    timestamp: Optional[datetime] = None
    image_metadata: Optional[Dict[str, Any]] = None
    processing_info: Optional[Dict[str, Any]] = None
    detection_summary: Optional[Dict[str, Any]] = None
    people_analysis: Optional[List[Dict[str, Any]]] = None
    scene_analysis: Optional[Dict[str, Any]] = None


class IAnalysisRepository(ABC):
    """Abstract repository interface for CV analysis data"""
    
    @abstractmethod
    async def save_analysis(self, analysis: CVAnalysisEntity) -> str:
        """Save analysis data and return the ID"""
        pass
    
    @abstractmethod
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[CVAnalysisEntity]:
        """Retrieve analysis by ID"""
        pass
    
    @abstractmethod
    async def get_all_analyses(self, limit: Optional[int] = None) -> List[CVAnalysisEntity]:
        """Retrieve all analyses with optional limit"""
        pass
    
    @abstractmethod
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis by ID"""
        pass
    
    @abstractmethod
    async def get_analyses_by_filter(self, filters: Dict[str, Any]) -> List[CVAnalysisEntity]:
        """Get analyses by custom filters"""
        pass


class IImageProcessor(ABC):
    """Abstract interface for image processing services"""
    
    @abstractmethod
    async def analyze_image(self, image_data: str) -> Dict[str, Any]:
        """Process image and return analysis results"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Get processor capabilities"""
        pass


class IConfigurationService(ABC):
    """Abstract interface for configuration management"""
    
    @abstractmethod
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        pass
    
    @abstractmethod
    def get_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration"""
        pass
    
    @abstractmethod
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        pass


class IDependencyContainer(ABC):
    """Abstract dependency injection container"""
    
    @abstractmethod
    def register_singleton(self, interface_type: type, implementation: Any) -> None:
        """Register a singleton service"""
        pass
    
    @abstractmethod
    def register_transient(self, interface_type: type, implementation_factory: callable) -> None:
        """Register a transient service"""
        pass
    
    @abstractmethod
    def resolve(self, interface_type: type) -> Any:
        """Resolve a service by type"""
        pass 