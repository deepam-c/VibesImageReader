"""
Dependency injection container implementation
"""

from typing import Dict, Any, Type, Callable
from core.interfaces import IDependencyContainer
import logging

logger = logging.getLogger(__name__)


class DependencyContainer(IDependencyContainer):
    """Simple dependency injection container"""
    
    def __init__(self):
        self._singletons: Dict[Type, Any] = {}
        self._transients: Dict[Type, Callable] = {}
        self._instances: Dict[Type, Any] = {}
    
    def register_singleton(self, interface_type: Type, implementation: Any) -> None:
        """Register a singleton service"""
        logger.info(f"Registering singleton: {interface_type.__name__} -> {type(implementation).__name__}")
        self._singletons[interface_type] = implementation
        
        # If it's already instantiated, store the instance
        if not callable(implementation):
            self._instances[interface_type] = implementation
    
    def register_transient(self, interface_type: Type, implementation_factory: Callable) -> None:
        """Register a transient service factory"""
        logger.info(f"Registering transient: {interface_type.__name__}")
        self._transients[interface_type] = implementation_factory
    
    def resolve(self, interface_type: Type) -> Any:
        """Resolve a service by type"""
        # Check if it's already instantiated
        if interface_type in self._instances:
            return self._instances[interface_type]
        
        # Check singletons
        if interface_type in self._singletons:
            implementation = self._singletons[interface_type]
            if callable(implementation):
                # Instantiate if it's a class/factory
                instance = implementation(self)
                self._instances[interface_type] = instance
                return instance
            else:
                # Return the pre-instantiated object
                return implementation
        
        # Check transients
        if interface_type in self._transients:
            factory = self._transients[interface_type]
            return factory(self)
        
        raise ValueError(f"Service not registered: {interface_type.__name__}")
    
    def register_instance(self, interface_type: Type, instance: Any) -> None:
        """Register a pre-created instance"""
        logger.info(f"Registering instance: {interface_type.__name__}")
        self._instances[interface_type] = instance


class ServiceLocator:
    """Global service locator for easy access"""
    
    _container: DependencyContainer = None
    
    @classmethod
    def initialize(cls, container: DependencyContainer) -> None:
        """Initialize the service locator"""
        cls._container = container
    
    @classmethod
    def get(cls, interface_type: Type) -> Any:
        """Get service from container"""
        if cls._container is None:
            raise RuntimeError("ServiceLocator not initialized")
        return cls._container.resolve(interface_type)
    
    @classmethod
    def register_singleton(cls, interface_type: Type, implementation: Any) -> None:
        """Register singleton service"""
        if cls._container is None:
            raise RuntimeError("ServiceLocator not initialized")
        cls._container.register_singleton(interface_type, implementation)
    
    @classmethod
    def register_transient(cls, interface_type: Type, implementation_factory: Callable) -> None:
        """Register transient service"""
        if cls._container is None:
            raise RuntimeError("ServiceLocator not initialized")
        cls._container.register_transient(interface_type, implementation_factory) 