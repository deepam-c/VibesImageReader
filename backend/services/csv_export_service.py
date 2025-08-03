"""
CSV Export Service Implementation
Follows Single Responsibility Principle - only handles CSV export operations
"""

import csv
import io
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

from core.interfaces import ICSVExporter

logger = logging.getLogger(__name__)


class CSVExporter(ICSVExporter):
    """
    Concrete implementation of CSV export functionality
    Follows Single Responsibility Principle - only handles CSV operations
    """
    
    def __init__(self):
        self.content_type = 'text/csv'
        self.file_extension = '.csv'
    
    def export(self, data: List[Dict[str, Any]], **kwargs) -> bytes:
        """Export data to CSV format"""
        columns = kwargs.get('columns', None)
        return self.export_to_csv(data, columns)
    
    def export_to_csv(self, data: List[Dict[str, Any]], columns: Optional[List[str]] = None) -> bytes:
        """
        Export analysis data to CSV format
        
        Args:
            data: List of analysis dictionaries
            columns: Optional list of specific columns to include
            
        Returns:
            CSV data as bytes
        """
        if not data:
            logger.warning("No data provided for CSV export")
            return b""
        
        try:
            # Create string buffer
            output = io.StringIO()
            
            # Flatten the data for CSV export
            flattened_data = [self._flatten_analysis_data(item) for item in data]
            
            if not flattened_data:
                return b""
            
            # Determine columns
            if columns:
                fieldnames = columns
            else:
                # Get all unique keys from flattened data
                fieldnames = set()
                for item in flattened_data:
                    fieldnames.update(item.keys())
                fieldnames = sorted(list(fieldnames))
            
            # Write CSV
            writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(flattened_data)
            
            # Convert to bytes
            csv_content = output.getvalue()
            output.close()
            
            logger.info(f"Successfully exported {len(flattened_data)} records to CSV")
            return csv_content.encode('utf-8')
            
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def get_content_type(self) -> str:
        """Get CSV MIME content type"""
        return self.content_type
    
    def get_file_extension(self) -> str:
        """Get CSV file extension"""
        return self.file_extension
    
    def _flatten_analysis_data(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Flatten nested analysis data for CSV export
        Converts nested dictionaries to flat structure with dot notation
        """
        flattened = {}
        
        try:
            # Basic information
            flattened['id'] = analysis.get('id', '')
            flattened['timestamp'] = self._format_timestamp(analysis.get('timestamp'))
            
            # Image metadata
            image_meta = analysis.get('imageMetadata', {})
            flattened['image_size'] = image_meta.get('size', 0)
            flattened['image_type'] = image_meta.get('type', '')
            flattened['image_width'] = image_meta.get('dimensions', {}).get('width', 0)
            flattened['image_height'] = image_meta.get('dimensions', {}).get('height', 0)
            
            # Processing info
            processing = analysis.get('processingInfo', {})
            flattened['processing_time_ms'] = processing.get('processingTime', 0)
            flattened['model_version'] = processing.get('modelVersion', '')
            flattened['processing_confidence'] = processing.get('confidence', 0)
            
            # Detection summary
            detection = analysis.get('detectionSummary', {})
            flattened['people_detected'] = detection.get('peopleDetected', 0)
            flattened['faces_analyzed'] = detection.get('facesAnalyzed', 0)
            flattened['average_confidence'] = detection.get('averageConfidence', 0)
            
            # Scene analysis
            scene = analysis.get('sceneAnalysis', {})
            flattened['scene_lighting'] = scene.get('lighting', '')
            flattened['scene_setting'] = scene.get('setting', '')
            flattened['image_quality'] = scene.get('imageQuality', '')
            
            # People analysis (aggregate data)
            people = analysis.get('peopleAnalysis', [])
            if people:
                # Count demographics
                ages = [p.get('demographics', {}).get('estimatedAge') for p in people if p.get('demographics', {}).get('estimatedAge')]
                genders = [p.get('demographics', {}).get('gender') for p in people if p.get('demographics', {}).get('gender')]
                emotions = [p.get('emotions', {}).get('primary') for p in people if p.get('emotions', {}).get('primary')]
                
                flattened['avg_estimated_age'] = sum(ages) / len(ages) if ages else 0
                flattened['gender_distribution'] = ', '.join(set(genders)) if genders else ''
                flattened['emotion_distribution'] = ', '.join(set(emotions)) if emotions else ''
                
                # Clothing and accessories (from first person as sample)
                first_person = people[0]
                appearance = first_person.get('appearance', {})
                clothing = appearance.get('clothing', {})
                
                flattened['detected_clothing'] = ', '.join(clothing.get('detected_items', []))
                flattened['clothing_colors'] = ', '.join(clothing.get('dominant_colors', []))
                flattened['clothing_style'] = clothing.get('style_category', '')
                flattened['accessories'] = ', '.join(appearance.get('accessories', []))
                flattened['overall_style'] = appearance.get('overall_style', '')
                flattened['outfit_formality'] = appearance.get('outfit_formality', '')
            
            return flattened
            
        except Exception as e:
            logger.warning(f"Error flattening analysis data: {str(e)}")
            # Return basic data even if flattening fails
            return {
                'id': analysis.get('id', ''),
                'timestamp': self._format_timestamp(analysis.get('timestamp')),
                'error': f'Data flattening error: {str(e)}'
            }
    
    def _format_timestamp(self, timestamp) -> str:
        """Format timestamp for CSV"""
        if not timestamp:
            return ''
        
        try:
            if hasattr(timestamp, 'toDate'):
                # Firestore timestamp
                return timestamp.toDate().strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(timestamp, datetime):
                return timestamp.strftime('%Y-%m-%d %H:%M:%S')
            else:
                # Try parsing as string
                dt = datetime.fromisoformat(str(timestamp).replace('Z', '+00:00'))
                return dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            return str(timestamp) 