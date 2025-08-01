# Refactored Computer Vision Backend

A modular, database-agnostic computer vision backend built with dependency injection and repository pattern.

## 🏗️ Architecture Overview

### Key Design Principles
- **Dependency Injection**: Loose coupling between components
- **Repository Pattern**: Database-agnostic data access
- **Interface Segregation**: Clear contracts between layers
- **Single Responsibility**: Each class has one job
- **Open/Closed Principle**: Easy to extend, difficult to break

### Layer Structure
```
┌─────────────────────┐
│   API Layer (Flask) │
├─────────────────────┤
│   Service Layer     │
├─────────────────────┤
│   Repository Layer  │
├─────────────────────┤
│   Infrastructure    │
└─────────────────────┘
```

## 🔧 Supported Databases

### Firebase (Default)
```bash
DATABASE_TYPE=firebase
FIREBASE_PROJECT_ID=your-project-id
```

### SQL Databases
```bash
DATABASE_TYPE=sql
SQL_DATABASE_URL=sqlite:///./cv_analytics.db
# or PostgreSQL: postgresql://user:password@localhost/database
# or MySQL: mysql+pymysql://user:password@localhost/database
```

### MongoDB
```bash
DATABASE_TYPE=mongodb
MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
MONGODB_DATABASE_NAME=cv_analytics
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_refactored.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your configuration
```

### 3. Run Application
```bash
python app_refactored.py
```

## 📦 Component Overview

### Core Interfaces (`core/interfaces.py`)
- `IAnalysisRepository`: Database operations contract
- `IImageProcessor`: Image analysis contract  
- `IConfigurationService`: Configuration management
- `IDependencyContainer`: Dependency injection contract

### Repository Implementations
- `FirebaseAnalysisRepository`: Firestore integration
- `SQLAnalysisRepository`: SQLAlchemy-based SQL support
- `MongoDBAnalysisRepository`: MongoDB integration

### Services
- `AnalysisService`: Business logic layer
- `SmartImageProcessor`: Computer vision processing
- `AppConfigurationService`: Configuration management

## 🔄 Adding a New Database

### 1. Create Repository Implementation
```python
# infrastructure/repositories/new_db_repository.py
from core.interfaces import IAnalysisRepository, CVAnalysisEntity

class NewDBRepository(IAnalysisRepository):
    async def save_analysis(self, analysis: CVAnalysisEntity) -> str:
        # Implementation here
        pass
    
    # Implement other interface methods...
```

### 2. Register in Factory
```python
# app_refactored.py
@staticmethod
def _create_repository(config: Dict[str, Any]) -> IAnalysisRepository:
    database_type = config.get('type', 'firebase')
    
    if database_type == 'newdb':
        return NewDBRepository(config)
    # ... existing implementations
```

### 3. Add Configuration
```python
# config/app_config.py
def get_database_config(self) -> Dict[str, Any]:
    if database_type == 'newdb':
        return {
            'type': 'newdb',
            'connection_string': os.getenv('NEWDB_CONNECTION')
        }
```

## 📊 API Endpoints

### Core Endpoints
- `POST /analyze-image` - Analyze and save image
- `GET /analyses` - Get all analyses
- `GET /analyses/{id}` - Get specific analysis
- `DELETE /analyses/{id}` - Delete analysis
- `GET /statistics` - Get system statistics
- `GET /capabilities` - Get system capabilities
- `GET /health` - Health check

### Example Usage
```bash
# Analyze image
curl -X POST http://localhost:5000/analyze-image \
  -H "Content-Type: application/json" \
  -d '{"image": "data:image/jpeg;base64,..."}'

# Get statistics
curl http://localhost:5000/statistics
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Database-Specific Tests
```bash
# Test Firebase
DATABASE_TYPE=firebase pytest tests/

# Test SQL
DATABASE_TYPE=sql pytest tests/

# Test MongoDB  
DATABASE_TYPE=mongodb pytest tests/
```

## 🔧 Configuration Options

### Environment Variables
| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_TYPE` | Database type (firebase/sql/mongodb) | firebase |
| `HOST` | Server host | 0.0.0.0 |
| `PORT` | Server port | 5000 |
| `DEBUG` | Debug mode | true |
| `MAX_IMAGE_SIZE` | Max image size in bytes | 10MB |

### Database Switching
Change databases without code changes:
```bash
# Switch to SQL
export DATABASE_TYPE=sql
export SQL_DATABASE_URL=postgresql://localhost/mydb

# Switch to MongoDB
export DATABASE_TYPE=mongodb
export MONGODB_CONNECTION_STRING=mongodb://localhost:27017/
```

## 🎯 Benefits

### For Developers
- **Easy Testing**: Mock any component
- **Clear Structure**: Obvious where to add features
- **Type Safety**: Interface contracts prevent errors
- **Database Freedom**: Switch databases anytime

### For Deployment
- **Environment Flexibility**: Same code, different databases
- **Scalability**: Easy to add caching, queuing
- **Monitoring**: Centralized logging and metrics
- **Cloud Ready**: Works with any cloud provider

## 🚧 Extension Points

### Adding New Features
1. **New Analysis Types**: Extend `IImageProcessor`
2. **New Storage**: Implement `IAnalysisRepository`
3. **New Configurations**: Extend `IConfigurationService`
4. **New Services**: Add to dependency container

### Middleware Integration
```python
# Add caching
container.register_singleton(ICacheService, RedisCacheService())

# Add message queue
container.register_singleton(IMessageQueue, RabbitMQService())
```

## 📈 Performance Considerations

### Async Operations
All repository operations are async for better performance:
```python
# Non-blocking database operations
result = await repository.save_analysis(analysis)
```

### Connection Pooling
Each repository manages its own connection pool:
```python
# SQL: SQLAlchemy handles connection pooling
# MongoDB: PyMongo handles connection pooling  
# Firebase: SDK handles connection management
```

### Caching Strategy
Easy to add caching at any layer:
```python
@cached(ttl=300)
async def get_analysis_by_id(self, analysis_id: str):
    return await self.repository.get_analysis_by_id(analysis_id)
```

## 🔒 Security

### Database Security
- Firebase: IAM rules and service accounts
- SQL: Connection encryption and user permissions
- MongoDB: Authentication and role-based access

### API Security
- CORS configuration
- Request validation
- Error message sanitization

## 📝 Logging

Structured logging throughout the application:
```python
logger.info("Analysis saved", extra={
    "analysis_id": analysis_id,
    "database_type": self.config["type"],
    "processing_time": processing_time
})
```

This refactored architecture provides a solid foundation for a scalable, maintainable computer vision backend that can adapt to changing requirements and different deployment environments. 