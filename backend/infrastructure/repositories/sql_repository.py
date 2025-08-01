"""
SQL database implementation of the analysis repository
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from core.interfaces import IAnalysisRepository, CVAnalysisEntity

logger = logging.getLogger(__name__)


class SQLAnalysisRepository(IAnalysisRepository):
    """SQL implementation of analysis repository using SQLAlchemy"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.engine = None
        self.Session = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQL database connection"""
        try:
            from sqlalchemy import create_engine, Column, String, DateTime, Text, Integer
            from sqlalchemy.ext.declarative import declarative_base
            from sqlalchemy.orm import sessionmaker
            
            # Create engine
            database_url = self.config.get('database_url', 'sqlite:///./cv_analytics.db')
            self.engine = create_engine(database_url)
            
            # Create session factory
            self.Session = sessionmaker(bind=self.engine)
            
            # Create tables
            self._create_tables()
            
            logger.info(f"SQL database initialized: {database_url}")
            
        except ImportError:
            logger.error("SQLAlchemy not installed. Install with: pip install sqlalchemy")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize SQL database: {e}")
            raise
    
    def _create_tables(self):
        """Create database tables"""
        from sqlalchemy import Column, String, DateTime, Text, Integer
        from sqlalchemy.ext.declarative import declarative_base
        
        Base = declarative_base()
        
        class CVAnalysis(Base):
            __tablename__ = 'cv_analyses'
            
            id = Column(String, primary_key=True)
            timestamp = Column(DateTime, default=datetime.now)
            image_metadata = Column(Text)  # JSON string
            processing_info = Column(Text)  # JSON string
            detection_summary = Column(Text)  # JSON string
            people_analysis = Column(Text)  # JSON string
            scene_analysis = Column(Text)  # JSON string
        
        self.CVAnalysis = CVAnalysis
        Base.metadata.create_all(self.engine)
        logger.info("SQL tables created successfully")
    
    async def save_analysis(self, analysis: CVAnalysisEntity) -> str:
        """Save analysis to SQL database"""
        try:
            import uuid
            
            session = self.Session()
            
            # Generate ID if not provided
            analysis_id = analysis.id or str(uuid.uuid4())
            
            # Create database record
            db_analysis = self.CVAnalysis(
                id=analysis_id,
                timestamp=analysis.timestamp or datetime.now(),
                image_metadata=json.dumps(analysis.image_metadata) if analysis.image_metadata else None,
                processing_info=json.dumps(analysis.processing_info) if analysis.processing_info else None,
                detection_summary=json.dumps(analysis.detection_summary) if analysis.detection_summary else None,
                people_analysis=json.dumps(analysis.people_analysis) if analysis.people_analysis else None,
                scene_analysis=json.dumps(analysis.scene_analysis) if analysis.scene_analysis else None
            )
            
            session.add(db_analysis)
            session.commit()
            session.close()
            
            logger.info(f"Analysis saved to SQL database with ID: {analysis_id}")
            return analysis_id
            
        except Exception as e:
            logger.error(f"Error saving analysis to SQL database: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            raise
    
    async def get_analysis_by_id(self, analysis_id: str) -> Optional[CVAnalysisEntity]:
        """Get analysis by ID from SQL database"""
        try:
            session = self.Session()
            
            db_analysis = session.query(self.CVAnalysis).filter(
                self.CVAnalysis.id == analysis_id
            ).first()
            
            session.close()
            
            if not db_analysis:
                return None
            
            return self._db_record_to_entity(db_analysis)
            
        except Exception as e:
            logger.error(f"Error getting analysis from SQL database: {e}")
            if 'session' in locals():
                session.close()
            raise
    
    async def get_all_analyses(self, limit: Optional[int] = None) -> List[CVAnalysisEntity]:
        """Get all analyses from SQL database"""
        try:
            session = self.Session()
            
            query = session.query(self.CVAnalysis).order_by(
                self.CVAnalysis.timestamp.desc()
            )
            
            if limit:
                query = query.limit(limit)
            
            db_analyses = query.all()
            session.close()
            
            analyses = [self._db_record_to_entity(db_analysis) for db_analysis in db_analyses]
            
            logger.info(f"Retrieved {len(analyses)} analyses from SQL database")
            return analyses
            
        except Exception as e:
            logger.error(f"Error getting all analyses from SQL database: {e}")
            if 'session' in locals():
                session.close()
            raise
    
    async def delete_analysis(self, analysis_id: str) -> bool:
        """Delete analysis from SQL database"""
        try:
            session = self.Session()
            
            result = session.query(self.CVAnalysis).filter(
                self.CVAnalysis.id == analysis_id
            ).delete()
            
            session.commit()
            session.close()
            
            success = result > 0
            if success:
                logger.info(f"Analysis {analysis_id} deleted from SQL database")
            else:
                logger.warning(f"Analysis {analysis_id} not found for deletion")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deleting analysis from SQL database: {e}")
            if 'session' in locals():
                session.rollback()
                session.close()
            return False
    
    async def get_analyses_by_filter(self, filters: Dict[str, Any]) -> List[CVAnalysisEntity]:
        """Get analyses by custom filters"""
        try:
            session = self.Session()
            
            query = session.query(self.CVAnalysis)
            
            # Apply simple filters (this is basic - could be extended for complex filtering)
            for field, value in filters.items():
                if hasattr(self.CVAnalysis, field):
                    query = query.filter(getattr(self.CVAnalysis, field) == value)
            
            db_analyses = query.all()
            session.close()
            
            analyses = [self._db_record_to_entity(db_analysis) for db_analysis in db_analyses]
            return analyses
            
        except Exception as e:
            logger.error(f"Error filtering analyses in SQL database: {e}")
            if 'session' in locals():
                session.close()
            raise
    
    def _db_record_to_entity(self, db_record) -> CVAnalysisEntity:
        """Convert database record to entity"""
        return CVAnalysisEntity(
            id=db_record.id,
            timestamp=db_record.timestamp,
            image_metadata=json.loads(db_record.image_metadata) if db_record.image_metadata else None,
            processing_info=json.loads(db_record.processing_info) if db_record.processing_info else None,
            detection_summary=json.loads(db_record.detection_summary) if db_record.detection_summary else None,
            people_analysis=json.loads(db_record.people_analysis) if db_record.people_analysis else None,
            scene_analysis=json.loads(db_record.scene_analysis) if db_record.scene_analysis else None
        ) 