"""
Data Export Service Implementation
Follows SOLID principles:
- Single Responsibility: Coordinates export operations
- Open/Closed: Extensible for new export formats
- Dependency Inversion: Depends on abstractions, not concrete classes
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import logging

from core.interfaces import IExportService, IDataExporter
from services.csv_export_service import CSVExporter

logger = logging.getLogger(__name__)


class DataExportService(IExportService):
    """
    Main export service that coordinates different export formats
    Follows Dependency Inversion Principle - depends on abstractions
    Follows Open/Closed Principle - extensible for new formats
    """
    
    def __init__(self, data_repository=None):
        """
        Initialize with dependency injection
        
        Args:
            data_repository: Repository for fetching analysis data
        """
        self.data_repository = data_repository
        self._exporters: Dict[str, IDataExporter] = {}
        self._register_default_exporters()
    
    def _register_default_exporters(self):
        """Register default export formats"""
        self.register_exporter('csv', CSVExporter())
        # Future formats can be added here without modifying existing code
        # self.register_exporter('json', JSONExporter())
        # self.register_exporter('xlsx', ExcelExporter())
    
    def register_exporter(self, format_name: str, exporter: IDataExporter):
        """
        Register a new export format (Open/Closed Principle)
        
        Args:
            format_name: Name of the export format
            exporter: Exporter implementation
        """
        self._exporters[format_name.lower()] = exporter
        logger.info(f"Registered exporter for format: {format_name}")
    
    def export_analyses(self, format_type: str, **kwargs) -> Tuple[bytes, str, str]:
        """
        Export analysis data in specified format
        
        Args:
            format_type: Export format ('csv', 'json', etc.)
            **kwargs: Additional export parameters
            
        Returns:
            Tuple of (data_bytes, content_type, filename)
        """
        format_type = format_type.lower()
        
        if format_type not in self._exporters:
            raise ValueError(f"Unsupported export format: {format_type}. Supported: {list(self._exporters.keys())}")
        
        try:
            # Get data (would normally come from repository)
            data = self._get_analysis_data(**kwargs)
            
            if not data:
                logger.warning("No data available for export")
                # Return empty file
                exporter = self._exporters[format_type]
                return b"", exporter.get_content_type(), self._generate_filename(format_type)
            
            # Export using appropriate exporter
            exporter = self._exporters[format_type]
            exported_data = exporter.export(data, **kwargs)
            
            # Generate filename
            filename = self._generate_filename(format_type)
            
            logger.info(f"Successfully exported {len(data)} records as {format_type}")
            return exported_data, exporter.get_content_type(), filename
            
        except Exception as e:
            logger.error(f"Error exporting data as {format_type}: {str(e)}")
            raise
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return list(self._exporters.keys())
    
    def _get_analysis_data(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Get analysis data for export
        This would typically fetch from a repository/database
        
        For now, returns mock data since we don't have direct database access
        """
        # In a real implementation, this would use the injected repository
        # return self.data_repository.get_all_analyses(**kwargs)
        
        # Mock data for demonstration
        return self._get_mock_data()
    
    def _get_mock_data(self) -> List[Dict[str, Any]]:
        """
        Temporary mock data for testing
        In production, this would be replaced with actual data fetching
        """
        return [
            {
                'id': 'analysis_001',
                'timestamp': datetime.now(),
                'imageMetadata': {
                    'size': 1024000,
                    'type': 'image/jpeg',
                    'dimensions': {'width': 1920, 'height': 1080}
                },
                'processingInfo': {
                    'processingTime': 1250,
                    'modelVersion': 'Smart CV v2.1',
                    'confidence': 0.85
                },
                'detectionSummary': {
                    'peopleDetected': 2,
                    'facesAnalyzed': 2,
                    'averageConfidence': 0.87
                },
                'sceneAnalysis': {
                    'lighting': 'natural',
                    'setting': 'outdoor',
                    'imageQuality': 'good'
                },
                'peopleAnalysis': [
                    {
                        'demographics': {'estimatedAge': 25, 'gender': 'female'},
                        'emotions': {'primary': 'happy'},
                        'appearance': {
                            'clothing': {
                                'detected_items': ['dress', 'shoes'],
                                'dominant_colors': ['blue', 'white'],
                                'style_category': 'casual'
                            },
                            'accessories': ['sunglasses'],
                            'overall_style': 'casual',
                            'outfit_formality': 'casual'
                        }
                    },
                    {
                        'demographics': {'estimatedAge': 30, 'gender': 'male'},
                        'emotions': {'primary': 'neutral'},
                        'appearance': {
                            'clothing': {
                                'detected_items': ['shirt', 'pants'],
                                'dominant_colors': ['black', 'gray'],
                                'style_category': 'business'
                            },
                            'accessories': [],
                            'overall_style': 'business',
                            'outfit_formality': 'formal'
                        }
                    }
                ]
            }
        ]
    
    def _generate_filename(self, format_type: str) -> str:
        """Generate filename for export"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exporter = self._exporters[format_type]
        extension = exporter.get_file_extension()
        return f"cv_analysis_export_{timestamp}{extension}"


class FirebaseDataExportService(DataExportService):
    """
    Extended export service that integrates with Firebase
    Demonstrates Open/Closed Principle - extending without modifying base class
    """
    
    def __init__(self, firebase_client=None):
        super().__init__()
        self.firebase_client = firebase_client
    
    def _get_analysis_data(self, **kwargs) -> List[Dict[str, Any]]:
        """Get actual data from Firebase"""
        if not self.firebase_client:
            logger.warning("Firebase client not available, using mock data")
            return super()._get_analysis_data(**kwargs)
        
        try:
            # This would integrate with the Firebase service
            # For now, return mock data
            return super()._get_analysis_data(**kwargs)
        except Exception as e:
            logger.error(f"Error fetching data from Firebase: {str(e)}")
            return [] 