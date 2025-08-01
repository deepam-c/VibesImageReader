"""
Application configuration service
"""

import os
from typing import Dict, Any
from core.interfaces import IConfigurationService


class AppConfigurationService(IConfigurationService):
    """Configuration service implementation"""
    
    def __init__(self):
        self.config = self._load_configuration()
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        database_type = os.getenv('DATABASE_TYPE', 'firebase')
        
        if database_type == 'firebase':
            return {
                'type': 'firebase',
                'service_account_path': os.getenv('FIREBASE_SERVICE_ACCOUNT_PATH'),
                'project_id': os.getenv('FIREBASE_PROJECT_ID'),
                'database_url': os.getenv('FIREBASE_DATABASE_URL')
            }
        elif database_type == 'sql':
            return {
                'type': 'sql',
                'database_url': os.getenv('SQL_DATABASE_URL', 'sqlite:///./cv_analytics.db')
            }
        elif database_type == 'mongodb':
            return {
                'type': 'mongodb',
                'connection_string': os.getenv('MONGODB_CONNECTION_STRING', 'mongodb://localhost:27017/'),
                'database_name': os.getenv('MONGODB_DATABASE_NAME', 'cv_analytics'),
                'collection_name': os.getenv('MONGODB_COLLECTION_NAME', 'analyses')
            }
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get image processing configuration"""
        return {
            'max_image_size': int(os.getenv('MAX_IMAGE_SIZE', 10 * 1024 * 1024)),  # 10MB
            'supported_formats': ['jpg', 'jpeg', 'png', 'webp'],
            'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
            'model_version': os.getenv('MODEL_VERSION', 'Smart CV v2.0')
        }
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return {
            'host': os.getenv('HOST', '0.0.0.0'),
            'port': int(os.getenv('PORT', 5000)),
            'debug': os.getenv('DEBUG', 'true').lower() == 'true',
            'cors_origins': os.getenv('CORS_ORIGINS', '*').split(','),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'workers': int(os.getenv('WORKERS', 1))
        }
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load all configuration"""
        return {
            'database': self.get_database_config(),
            'processing': self.get_processing_config(),
            'server': self.get_server_config()
        } 