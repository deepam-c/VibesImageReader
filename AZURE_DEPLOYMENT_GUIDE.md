# Azure Deployment Guide for Vibes Image Reader

This guide will help you deploy your Vibes Image Reader application to Microsoft Azure using Azure Container Apps for the backend and Azure Static Web Apps for the frontend.

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Azure Static  │    │  Azure Container │    │    Firebase     │
│    Web Apps     │───▶│      Apps        │───▶│   (Database)    │
│   (Frontend)    │    │   (Backend API)  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │
         │              ┌──────────────────┐
         └─────────────▶│ Azure Container  │
                        │    Registry      │
                        └──────────────────┘
```

## Prerequisites

Before starting the deployment, ensure you have:

1. **Azure CLI** installed and configured
2. **Docker** installed and running
3. **Node.js** 18+ and npm
4. **Git** for version control
5. An **Azure subscription** with appropriate permissions
6. **Firebase** project (if using Firebase for database)

### Install Azure CLI

```bash
# Windows (using PowerShell as Administrator)
Invoke-WebRequest -Uri https://aka.ms/installazurecliwindows -OutFile .\AzureCLI.msi; Start-Process msiexec.exe -Wait -ArgumentList '/I AzureCLI.msi /quiet'; rm .\AzureCLI.msi

# macOS
brew install azure-cli

# Linux (Ubuntu/Debian)
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
```

## Step 1: Prepare Your Application

### 1.1 Update Frontend Configuration

Your Next.js app is already configured for static export. Verify the configuration in `next.config.js`:

```javascript
const nextConfig = {
  output: 'export',
  trailingSlash: true,
  skipTrailingSlashRedirect: true,
  distDir: 'out',
  images: {
    unoptimized: true
  }
}
```

### 1.2 Environment Variables

Create a `.env.local` file in your project root:

```bash
cp .env.example .env.local
```

Fill in your actual values:
- Firebase configuration
- Backend API URL (will be provided after backend deployment)

## Step 2: Azure Login and Setup

```bash
# Login to Azure
az login

# Set your subscription (if you have multiple)
az account set --subscription "Your Subscription Name or ID"

# Verify you're logged in
az account show

# List your existing resource groups (if you already have one)
az group list --output table
```

## Step 3: Deploy Using Automated Script

### 3.1 Make the deployment script executable

```bash
chmod +x azure/deploy.sh
```

### 3.2 Run the deployment

If you already have a resource group, use it by setting the environment variable:

```bash
export RESOURCE_GROUP="your-existing-rg-name"
cd azure
./deploy.sh
```

Or with custom parameters:

```bash
./deploy.sh --environment prod --location eastus --resource-group your-existing-rg-name
```

If you don't have a resource group, the script will create one for you:

```bash
cd azure
./deploy.sh
```

The script will:
1. ✅ Check prerequisites
2. ✅ Create Azure resource group
3. ✅ Deploy infrastructure using Bicep templates
4. ✅ Build and push Docker images
5. ✅ Deploy the backend to Container Apps
6. ✅ Provide deployment URLs

## Step 4: Manual Deployment (Alternative)

If you prefer manual deployment or the script fails:

### 4.1 Create Resource Group (Skip if you already have one)

If you don't have a resource group yet:

```bash
az group create --name your-rg-name --location eastus
```

If you already have a resource group, just use its name in the following commands.

### 4.2 Deploy Infrastructure

Replace `your-rg-name` with your actual resource group name:

```bash
az deployment group create \
  --resource-group your-rg-name \
  --template-file azure/bicep/main.bicep \
  --parameters namePrefix=vibes environment=prod
```

### 4.3 Build and Push Backend Container

```bash
# Set your resource group name
RESOURCE_GROUP="your-rg-name"

# Get registry details
REGISTRY_NAME=$(az deployment group show --resource-group $RESOURCE_GROUP --name main --query "properties.outputs.containerRegistryName.value" -o tsv)

# Login to registry
az acr login --name $REGISTRY_NAME

# Build and push
cd backend
docker build -t $REGISTRY_NAME.azurecr.io/vibes-backend:latest .
docker push $REGISTRY_NAME.azurecr.io/vibes-backend:latest
```

## Step 5: Configure GitHub Actions (Recommended)

### 5.1 Create GitHub Secrets

Go to your GitHub repository → Settings → Secrets and variables → Actions, and add:

#### For Static Web Apps:
- `AZURE_STATIC_WEB_APPS_API_TOKEN`: Get from Azure Portal → Static Web Apps → Manage deployment token

#### For Container Apps:
- `AZURE_CREDENTIALS`: Service principal credentials
- `AZURE_REGISTRY_USERNAME`: Container registry username
- `AZURE_REGISTRY_PASSWORD`: Container registry password

#### For Frontend Environment:
- `NEXT_PUBLIC_FIREBASE_API_KEY`
- `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`
- `NEXT_PUBLIC_FIREBASE_PROJECT_ID`
- `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET`
- `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID`
- `NEXT_PUBLIC_FIREBASE_APP_ID`
- `NEXT_PUBLIC_API_URL`: Your backend URL from Step 3

### 5.2 Create Azure Service Principal

```bash
az ad sp create-for-rbac \
  --name "vibes-github-actions" \
  --role contributor \
  --scopes /subscriptions/{subscription-id}/resourceGroups/vibes-rg \
  --sdk-auth
```

Copy the JSON output to `AZURE_CREDENTIALS` secret.

### 5.3 Get Container Registry Credentials

```bash
az acr credential show --name vibesregistryprod
```

Use the username and password for GitHub secrets.

## Step 6: Configure Static Web App

### 6.1 Create Static Web App

```bash
az staticwebapp create \
  --name vibes-frontend-prod \
  --resource-group vibes-rg \
  --source https://github.com/deepam-c/VibesImageReader \
  --location "East US 2" \
  --branch master \
  --app-location "/" \
  --output-location "out"
```

### 6.2 Get Deployment Token

```bash
az staticwebapp secrets list --name vibes-frontend-prod --resource-group vibes-rg
```

## Step 7: Update Frontend API URL

After backend deployment, update your frontend to use the backend URL:

1. Update `.env.local` with the backend URL
2. Commit and push changes to trigger deployment

## Step 8: Verify Deployment

### 8.1 Check Backend Health

```bash
curl https://your-backend-url.azurecontainerapps.io/health
```

### 8.2 Check Frontend

Visit your Static Web App URL and test the camera capture functionality.

### 8.3 Monitor Applications

- **Azure Portal**: Monitor resource usage and logs
- **Application Insights**: Set up for detailed monitoring
- **GitHub Actions**: Check deployment status

## Troubleshooting

### Common Issues

1. **Docker Build Fails**
   ```bash
   # Clear Docker cache
   docker system prune -f
   
   # Check system requirements
   docker info
   ```

2. **Container Registry Access Denied**
   ```bash
   # Re-login to registry
   az acr login --name vibesregistryprod
   
   # Check permissions
   az role assignment list --assignee $(az account show --query user.name -o tsv)
   ```

3. **Static Web App Build Fails**
   - Check Node.js version in GitHub Actions
   - Verify all environment variables are set
   - Check build logs in GitHub Actions

4. **CORS Issues**
   - Ensure backend CORS is configured for your frontend domain
   - Check that API URLs are correctly configured

### Debugging Commands

```bash
# Check container app logs
az containerapp logs show --name vibes-backend-prod --resource-group vibes-rg

# Check static web app details
az staticwebapp show --name vibes-frontend-prod --resource-group vibes-rg

# List all resources
az resource list --resource-group vibes-rg --output table
```

## Cost Optimization

### Development Environment

For development/testing, you can use:
- **Container Apps**: Consumption plan (pay-per-use)
- **Static Web Apps**: Free tier
- **Container Registry**: Basic SKU

### Production Environment

Consider:
- **Azure Application Insights** for monitoring
- **Azure CDN** for global distribution
- **Custom domains** and SSL certificates
- **Auto-scaling** rules for Container Apps

## Security Best Practices

1. **Use Managed Identity** where possible
2. **Enable HTTPS only** for all services
3. **Configure proper CORS** policies
4. **Use Azure Key Vault** for sensitive configuration
5. **Enable Azure AD authentication** if needed
6. **Regular security updates** for container images

## Next Steps

1. **Set up monitoring** with Application Insights
2. **Configure custom domains** for production
3. **Implement CI/CD** for multiple environments
4. **Set up backup strategies** for data
5. **Performance optimization** based on usage patterns

## Support

If you encounter issues:
1. Check the [Azure Documentation](https://docs.microsoft.com/azure/)
2. Review GitHub Actions logs
3. Check Azure Portal for service health
4. Contact Azure Support if needed

---

**Deployment completed!** 🎉

Your application should now be running on:
- **Frontend**: `https://your-static-web-app.azurestaticapps.net`
- **Backend**: `https://your-container-app.azurecontainerapps.io` 

## 🔍 **Step 1: Check What Image is Currently Running**

In **Azure Cloud Shell**, let's see what's actually deployed:

```bash
<code_block_to_apply_changes_from>
```

## 🔧 **Step 2: Manually Update to Flask Image**

Let's manually update the container to use your Flask backend:

```bash
# Get the correct image tag
IMAGE_TAG="vibesregistryprods.azurecr.io/vibes-backend:8a2af75"

# Update container app to use Flask backend
az containerapp update \
  --name vibes-backend-prod \
  --resource-group vibesCognizant \
  --image $IMAGE_TAG
```

## 🔍 **Step 3: Verify the Flask Image Exists**

Let's check if your Flask image was actually built and pushed:

```bash
# List images in your registry
az acr repository list --name vibesregistryprods --output table

# Check specific image tags
az acr repository show-tags --name vibesregistryprods --repository vibes-backend --output table
```

## 🎯 **Most Likely Issues:**

### **Issue A: Image Build Failed**
- The **Docker build succeeded** but image wasn't properly tagged
- **Solution**: Check the exact image tag that was pushed

### **Issue B: Container App Didn't Update**
- The **registry set** command worked, but image update failed
- **Solution**: Force container app to use the new image

### **Issue C: Wrong Image Tag**
- **Container App** is looking for a different tag than what was pushed
- **Solution**: Use the exact image tag from the build

## 🚀 **Quick Fix Commands:**

**Run these in Azure Cloud Shell:**

```bash
# 1. Check what's currently running
az containerapp show --name vibes-backend-prod --resource-group vibesCognizant --query "properties.template.containers[0].image"

# 2. List available Flask images
az acr repository show-tags --name vibesregistryprods --repository vibes-backend --output table

# 3. Force update to latest Flask image (replace with actual tag)
az containerapp update \
  --name vibes-backend-prod \
  --resource-group vibesCognizant \
  --image vibesregistryprods.azurecr.io/vibes-backend:latest

# 4. Restart the container app
az containerapp restart \
  --name vibes-backend-prod \
  --resource-group vibesCognizant
```

## 📊 **Expected Result:**

After running these commands:
- **Welcome page should disappear**
- **`/health` should return**: `{"status": "healthy", "timestamp": "...", "service": "Human Analysis API"}`
- **`/analyze-image` should return**: Method not allowed (GET) or proper POST endpoint

## 🔍 **Let's Debug Step by Step:**

**Can you run the first command and tell me:**
1. **What image is currently running?** (from the first command)
2. **What Flask images are available?** (from the second command)

Once I know this, I can give you the exact command to fix it! 🔧

The issue is definitely that **nginx is still running instead of your Flask backend**. 

## 🔧 **Corrected Commands (Run these one by one):**

### **1. Check what's currently running:**
```bash
az containerapp show --name vibes-backend-prod --resource-group vibesCognizant --query "properties.template.containers[0].image"
```

### **2. List available Flask images:**
```bash
az acr repository list --name vibesregistryprods --output table
```

### **3. Check specific vibes-backend tags:**
```bash
az acr repository show-tags --name vibesregistryprods --repository vibes-backend --output table
```

### **4. Force update to Flask image (use the latest commit):**
```bash
az containerapp update \
  --name vibes-backend-prod \
  --resource-group vibesCognizant \
  --image vibesregistryprods.azurecr.io/vibes-backend:8a2af75
```

### **5. Alternative restart (if needed):**
```bash
# Instead of restart, use revision restart
az containerapp revision restart \
  --name vibes-backend-prod \
  --resource-group vibesCognizant \
  --revision $(az containerapp revision list --name vibes-backend-prod --resource-group vibesCognizant --query "[0].name" -o tsv)
```

## 🎯 **Or Simpler Approach:**

**Just run the update command with the specific image tag:**

```bash
az containerapp update \
  --name vibes-backend-prod \
  --resource-group vibesCognizant \
  --image vibesregistryprods.azurecr.io/vibes-backend:8a2af75
```

This should automatically restart the container with your Flask backend.

## 📊 **Let's Start Step by Step:**

**Run these commands one by one and tell me the output:**

1. **First, check what image is currently running:**
   ```bash
   az containerapp show --name vibes-backend-prod --resource-group vibesCognizant --query "properties.template.containers[0].image"
   ```

2. **Then check what Flask images are available:**
   ```bash
   az acr repository show-tags --name vibesregistryprods --repository vibes-backend --output table
   ```

Once I see these outputs, I'll give you the exact command to switch from nginx to your Flask backend! 🔧

The `restart` command issue is just a version compatibility problem - the update command will work fine. 