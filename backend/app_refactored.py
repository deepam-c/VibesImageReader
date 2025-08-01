"""
Refactored Flask application with dependency injection
"""

import asyncio
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from typing import Dict, Any

# Core interfaces
from core.interfaces import IAnalysisRepository, IImageProcessor, IConfigurationService

# Infrastructure
from infrastructure.dependency_container import DependencyContainer, ServiceLocator
from infrastructure.repositories.firebase_repository import FirebaseAnalysisRepository
from infrastructure.repositories.sql_repository import SQLAnalysisRepository
from infrastructure.repositories.mongodb_repository import MongoDBAnalysisRepository

# Services
from services.analysis_service import AnalysisService
from services.smart_image_processor import SmartImageProcessor

# Configuration
from config.app_config import AppConfigurationService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ApplicationFactory:
    """Factory for creating and configuring the application"""
    
    @staticmethod
    def create_app() -> Flask:
        """Create and configure Flask application with dependency injection"""
        
        # Create Flask app
        app = Flask(__name__)
        
        # Initialize dependency container
        container = DependencyContainer()
        ServiceLocator.initialize(container)
        
        # Configure dependencies
        ApplicationFactory._configure_dependencies(container)
        
        # Configure Flask app
        ApplicationFactory._configure_flask(app)
        
        # Register routes
        ApplicationFactory._register_routes(app)
        
        logger.info("🚀 Application created successfully with dependency injection")
        return app
    
    @staticmethod
    def _configure_dependencies(container: DependencyContainer):
        """Configure dependency injection"""
        
        # Register configuration service
        config_service = AppConfigurationService()
        container.register_instance(IConfigurationService, config_service)
        
        # Register image processor
        processing_config = config_service.get_processing_config()
        image_processor = SmartImageProcessor(processing_config)
        container.register_instance(IImageProcessor, image_processor)
        
        # Register repository based on configuration
        database_config = config_service.get_database_config()
        repository = ApplicationFactory._create_repository(database_config)
        container.register_instance(IAnalysisRepository, repository)
        
        # Register analysis service
        container.register_transient(
            AnalysisService,
            lambda c: AnalysisService(
                c.resolve(IAnalysisRepository),
                c.resolve(IImageProcessor)
            )
        )
        
        logger.info("✅ Dependencies configured successfully")
    
    @staticmethod
    def _create_repository(config: Dict[str, Any]) -> IAnalysisRepository:
        """Create repository based on configuration"""
        
        database_type = config.get('type', 'firebase')
        
        if database_type == 'firebase':
            logger.info("📄 Initializing Firebase repository")
            return FirebaseAnalysisRepository(config)
        elif database_type == 'sql':
            logger.info("🗄️ Initializing SQL repository")
            return SQLAnalysisRepository(config)
        elif database_type == 'mongodb':
            logger.info("🍃 Initializing MongoDB repository")
            return MongoDBAnalysisRepository(config)
        else:
            raise ValueError(f"Unsupported database type: {database_type}")
    
    @staticmethod
    def _configure_flask(app: Flask):
        """Configure Flask application"""
        
        config_service = ServiceLocator.get(IConfigurationService)
        server_config = config_service.get_server_config()
        
        # Enable CORS
        CORS(app, origins=server_config['cors_origins'])
        
        # Set configuration
        app.config['DEBUG'] = server_config['debug']
        
        logger.info("⚙️ Flask application configured")
    
    @staticmethod
    def _register_routes(app: Flask):
        """Register API routes"""
        
        @app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'version': '2.0.0'
            })
        
        @app.route('/analyze-image', methods=['POST'])
        def analyze_image():
            """Analyze image endpoint"""
            try:
                data = request.get_json()
                
                if not data or 'image' not in data:
                    return jsonify({'error': 'No image data provided'}), 400
                
                # Get analysis service
                analysis_service = ServiceLocator.get(AnalysisService)
                
                # Run async analysis
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        analysis_service.analyze_and_save_image(
                            data['image'],
                            data.get('metadata', {})
                        )
                    )
                finally:
                    loop.close()
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in analyze_image: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/analyses', methods=['GET'])
        def get_analyses():
            """Get all analyses"""
            try:
                limit = request.args.get('limit', type=int)
                
                analysis_service = ServiceLocator.get(AnalysisService)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        analysis_service.get_all_analyses(limit)
                    )
                finally:
                    loop.close()
                
                return jsonify({'analyses': result})
                
            except Exception as e:
                logger.error(f"Error in get_analyses: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/analyses/<analysis_id>', methods=['GET'])
        def get_analysis(analysis_id: str):
            """Get analysis by ID"""
            try:
                analysis_service = ServiceLocator.get(AnalysisService)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        analysis_service.get_analysis_by_id(analysis_id)
                    )
                finally:
                    loop.close()
                
                if result:
                    return jsonify(result)
                else:
                    return jsonify({'error': 'Analysis not found'}), 404
                
            except Exception as e:
                logger.error(f"Error in get_analysis: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/analyses/<analysis_id>', methods=['DELETE'])
        def delete_analysis(analysis_id: str):
            """Delete analysis by ID"""
            try:
                analysis_service = ServiceLocator.get(AnalysisService)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        analysis_service.delete_analysis(analysis_id)
                    )
                finally:
                    loop.close()
                
                if result:
                    return jsonify({'message': 'Analysis deleted successfully'})
                else:
                    return jsonify({'error': 'Analysis not found'}), 404
                
            except Exception as e:
                logger.error(f"Error in delete_analysis: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/statistics', methods=['GET'])
        def get_statistics():
            """Get analysis statistics"""
            try:
                analysis_service = ServiceLocator.get(AnalysisService)
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        analysis_service.get_statistics()
                    )
                finally:
                    loop.close()
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error in get_statistics: {e}")
                return jsonify({'error': str(e)}), 500
        
        @app.route('/capabilities', methods=['GET'])
        def get_capabilities():
            """Get system capabilities"""
            try:
                analysis_service = ServiceLocator.get(AnalysisService)
                capabilities = analysis_service.get_capabilities()
                
                return jsonify(capabilities)
                
            except Exception as e:
                logger.error(f"Error in get_capabilities: {e}")
                return jsonify({'error': str(e)}), 500
        
        logger.info("🛣️ API routes registered")


def create_app() -> Flask:
    """Application factory function"""
    return ApplicationFactory.create_app()


if __name__ == '__main__':
    app = create_app()
    
    # Get server configuration
    config_service = ServiceLocator.get(IConfigurationService)
    server_config = config_service.get_server_config()
    
    print("🚀 Starting Refactored Computer Vision API...")
    print("🧠 AI Models: Smart Feature Detection with Dependency Injection")
    print("🏗️ Architecture: Repository Pattern + Dependency Injection")
    print(f"📊 Database: {config_service.get_database_config()['type'].upper()}")
    print(f"✅ Server starting on http://localhost:{server_config['port']}")
    
    app.run(
        host=server_config['host'],
        port=server_config['port'],
        debug=server_config['debug']
    ) 