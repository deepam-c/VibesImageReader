#!/bin/bash

# Azure Deployment Script for Vibes Image Reader Application
set -e

# Configuration
SUBSCRIPTION_ID="${AZURE_SUBSCRIPTION_ID}"
RESOURCE_GROUP="${RESOURCE_GROUP:-vibes-rg}"
LOCATION="${LOCATION:-eastus}"
ENVIRONMENT="${ENVIRONMENT:-prod}"
NAME_PREFIX="${NAME_PREFIX:-vibes}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check if Azure CLI is installed
    if ! command -v az &> /dev/null; then
        print_error "Azure CLI is not installed. Please install it first."
        exit 1
    fi
    
    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install it first."
        exit 1
    fi
    
    # Check if logged in to Azure
    if ! az account show &> /dev/null; then
        print_error "Please login to Azure CLI first: az login"
        exit 1
    fi
    
    print_success "All prerequisites checked"
}

# Set Azure subscription
set_subscription() {
    if [ -n "$SUBSCRIPTION_ID" ]; then
        print_status "Setting Azure subscription to $SUBSCRIPTION_ID"
        az account set --subscription "$SUBSCRIPTION_ID"
    else
        print_warning "AZURE_SUBSCRIPTION_ID not set, using current subscription"
    fi
}

# Create resource group (if it doesn't exist)
create_resource_group() {
    # Check if resource group already exists
    if az group show --name "$RESOURCE_GROUP" --output none 2>/dev/null; then
        print_warning "Resource group $RESOURCE_GROUP already exists, skipping creation"
    else
        print_status "Creating resource group $RESOURCE_GROUP in $LOCATION"
        az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output none
        print_success "Resource group created"
    fi
}

# Deploy infrastructure using Bicep
deploy_infrastructure() {
    print_status "Deploying Azure infrastructure..."
    
    DEPLOYMENT_NAME="vibes-deployment-$(date +%Y%m%d-%H%M%S)"
    
    az deployment group create \
        --resource-group "$RESOURCE_GROUP" \
        --template-file "bicep/main.bicep" \
        --parameters namePrefix="$NAME_PREFIX" environment="$ENVIRONMENT" location="$LOCATION" \
        --name "$DEPLOYMENT_NAME" \
        --output table
    
    print_success "Infrastructure deployed successfully"
}

# Get deployment outputs
get_deployment_outputs() {
    print_status "Getting deployment outputs..."
    
    # Get the latest deployment
    LATEST_DEPLOYMENT=$(az deployment group list --resource-group "$RESOURCE_GROUP" --query "[0].name" -o tsv)
    
    # Get outputs
    BACKEND_URL=$(az deployment group show --resource-group "$RESOURCE_GROUP" --name "$LATEST_DEPLOYMENT" --query "properties.outputs.backendUrl.value" -o tsv)
    FRONTEND_URL=$(az deployment group show --resource-group "$RESOURCE_GROUP" --name "$LATEST_DEPLOYMENT" --query "properties.outputs.frontendUrl.value" -o tsv)
    REGISTRY_LOGIN_SERVER=$(az deployment group show --resource-group "$RESOURCE_GROUP" --name "$LATEST_DEPLOYMENT" --query "properties.outputs.containerRegistryLoginServer.value" -o tsv)
    REGISTRY_NAME=$(az deployment group show --resource-group "$RESOURCE_GROUP" --name "$LATEST_DEPLOYMENT" --query "properties.outputs.containerRegistryName.value" -o tsv)
    
    echo "BACKEND_URL=$BACKEND_URL" > .env.azure
    echo "FRONTEND_URL=$FRONTEND_URL" >> .env.azure
    echo "REGISTRY_LOGIN_SERVER=$REGISTRY_LOGIN_SERVER" >> .env.azure
    echo "REGISTRY_NAME=$REGISTRY_NAME" >> .env.azure
    
    print_success "Deployment outputs saved to .env.azure"
}

# Build and push Docker image
build_and_push_image() {
    print_status "Building and pushing Docker image..."
    
    source .env.azure
    
    # Login to Azure Container Registry
    az acr login --name "$REGISTRY_NAME"
    
    # Build and push image
    cd backend
    docker build -t "$REGISTRY_LOGIN_SERVER/vibes-backend:latest" .
    docker push "$REGISTRY_LOGIN_SERVER/vibes-backend:latest"
    cd ..
    
    print_success "Docker image built and pushed"
}

# Update container app with new image
update_container_app() {
    print_status "Updating container app with new image..."
    
    source .env.azure
    
    CONTAINER_APP_NAME="${NAME_PREFIX}-backend-${ENVIRONMENT}"
    
    az containerapp update \
        --name "$CONTAINER_APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --image "$REGISTRY_LOGIN_SERVER/vibes-backend:latest" \
        --output table
    
    print_success "Container app updated"
}

# Main deployment function
main() {
    print_status "Starting Azure deployment for Vibes Image Reader"
    
    check_prerequisites
    set_subscription
    create_resource_group
    deploy_infrastructure
    get_deployment_outputs
    build_and_push_image
    update_container_app
    
    print_success "Deployment completed successfully!"
    print_status "Backend URL: $BACKEND_URL"
    print_status "Frontend URL: $FRONTEND_URL"
    print_status "Next steps:"
    echo "1. Configure GitHub secrets for automated deployments"
    echo "2. Update your frontend to use the backend URL"
    echo "3. Configure your domain and SSL certificates if needed"
}

# Help function
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -h, --help              Show this help message"
    echo "  -e, --environment ENV   Set environment (default: prod)"
    echo "  -l, --location LOC      Set Azure location (default: eastus)"
    echo "  -g, --resource-group RG Set resource group name (default: vibes-rg)"
    echo "  -p, --prefix PREFIX     Set name prefix (default: vibes)"
    echo ""
    echo "Environment variables:"
    echo "  AZURE_SUBSCRIPTION_ID   Azure subscription ID"
    echo "  RESOURCE_GROUP          Resource group name"
    echo "  LOCATION                Azure location"
    echo "  ENVIRONMENT             Environment name"
    echo "  NAME_PREFIX             Name prefix for resources"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -l|--location)
            LOCATION="$2"
            shift 2
            ;;
        -g|--resource-group)
            RESOURCE_GROUP="$2"
            shift 2
            ;;
        -p|--prefix)
            NAME_PREFIX="$2"
            shift 2
            ;;
        *)
            print_error "Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Run main function
main 