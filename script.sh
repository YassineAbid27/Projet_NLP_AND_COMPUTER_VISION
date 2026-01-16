#!/usr/bin/env bash
set -euo pipefail
#################################
# VARIABLES FOR SIGN LANGUAGE TRANSLATOR
#################################
RESOURCE_GROUP="rg-sign-language-translator"  
LOCATION="italynorth"   # Primary location
FALLBACK_LOCATION="northeurope"     # Fallback location
ACR_NAME="signlang$(whoami | tr '[:upper:]' '[:lower:]' | tr -cd '[:alnum:]')"  # 100% lowercase
CONTAINER_APP_NAME="sign-language-app" 
CONTAINERAPPS_ENV="env-sign-language"
IMAGE_NAME="sign-language-translator"
IMAGE_TAG="v1"
TARGET_PORT=8501  # Streamlit default port

#################################
# 0) Azure Context + Extensions Verification
#################################
echo "Checking Azure context..."
az account show --query "{name:name, cloudName:cloudName}" -o json >/dev/null

echo "Checking/installing Azure CLI extensions..."

# Check and install containerapp if necessary
if ! az extension show --name containerapp >/dev/null 2>&1; then
    echo "📦 Installing containerapp extension..."
    az extension add --name containerapp --upgrade -y --only-show-errors
    echo "✅ Extension containerapp installed"
else
    echo "✅ Extension containerapp already installed"
    # Silent update
    az extension update --name containerapp -y --only-show-errors 2>/dev/null || true
fi

# List installed extensions for verification
echo "Installed extensions:"
az extension list --query "[].{Name:name, Version:version}" -o table

#################################
# 1) Required Providers
#################################
echo "Registering providers..."
az provider register --namespace Microsoft.ContainerRegistry --wait
az provider register --namespace Microsoft.App --wait
az provider register --namespace Microsoft.Web --wait
az provider register --namespace Microsoft.OperationalInsights --wait

#################################
# 2) Resource Group
#################################
echo "Creating/validating resource group..."
az group create -n "$RESOURCE_GROUP" -l "$LOCATION" >/dev/null || true
echo "✅ Resource Group OK: $RESOURCE_GROUP"

#################################
# 3) Creating ACR (with verification)
#################################
echo "Creating Container Registry (ACR) in $LOCATION..."

# Preliminary verification
if [[ ! "$ACR_NAME" =~ ^[a-z0-9]{5,50}$ ]]; then
    echo "❌ ERROR: Invalid ACR name: $ACR_NAME"
    echo "   Must contain 5-50 alphanumeric characters in lowercase"
    exit 1
fi

echo "ACR name validated: $ACR_NAME (${#ACR_NAME} characters)"

set +e
az acr create \
  --resource-group "$RESOURCE_GROUP" \
  --name "$ACR_NAME" \
  --sku Basic \
  --admin-enabled true \
  --location "$LOCATION" >/dev/null 2>&1
ACR_RC=$?
set -e

if [ $ACR_RC -ne 0 ]; then
  echo "⚠️ ACR blocked in $LOCATION. Fallback => $FALLBACK_LOCATION"
  LOCATION="$FALLBACK_LOCATION"
  az acr create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$ACR_NAME" \
    --sku Basic \
    --admin-enabled true \
    --location "$LOCATION" >/dev/null
fi

# Wait for complete creation
sleep 5
echo "✅ ACR created: $ACR_NAME (region=$LOCATION)"

#################################
# 4) Login ACR + Push image
#################################
echo "Connecting to registry..."
az acr login --name "$ACR_NAME" >/dev/null

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
echo "ACR_LOGIN_SERVER=$ACR_LOGIN_SERVER"

# Get credentials
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')
IMAGE="$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "Building + Tagging + Pushing..."
docker build -t "$IMAGE_NAME:$IMAGE_TAG" .
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker tag "$IMAGE_NAME:$IMAGE_TAG" "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:$IMAGE_TAG"
docker push "$ACR_LOGIN_SERVER/$IMAGE_NAME:latest"
echo "✅ Image pushed to ACR"

#################################
# 5) Log Analytics
#################################
LAW_NAME="law-sign-language-$(whoami)-$RANDOM"
echo "Creating Log Analytics: $LAW_NAME"
az monitor log-analytics workspace create -g "$RESOURCE_GROUP" -n "$LAW_NAME" -l "$LOCATION" >/dev/null
sleep 10  # Necessary wait

# Corrected command with explicit parameters
LAW_ID=$(az monitor log-analytics workspace show \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LAW_NAME" \
    --query customerId -o tsv | tr -d '\r')

LAW_KEY=$(az monitor log-analytics workspace get-shared-keys \
    --resource-group "$RESOURCE_GROUP" \
    --workspace-name "$LAW_NAME" \
    --query primarySharedKey -o tsv | tr -d '\r')
echo "✅ Log Analytics OK"

#################################
# 6) Container Apps Environment
#################################
echo "Creating/validating Container Apps Environment: $CONTAINERAPPS_ENV"
if ! az containerapp env show -n "$CONTAINERAPPS_ENV" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp env create \
    -n "$CONTAINERAPPS_ENV" \
    -g "$RESOURCE_GROUP" \
    -l "$LOCATION" \
    --logs-workspace-id "$LAW_ID" \
    --logs-workspace-key "$LAW_KEY" >/dev/null
fi
echo "✅ Environment OK"

#################################
# 7) Deploying Container App
#################################
echo "Deploying Container App: $CONTAINER_APP_NAME"
if az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" >/dev/null 2>&1; then
  az containerapp update \
    -n "$CONTAINER_APP_NAME" \
    -g "$RESOURCE_GROUP" \
    --image "$IMAGE" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" >/dev/null
else
  az containerapp create \
    -n "$CONTAINER_APP_NAME" \
    -g "$RESOURCE_GROUP" \
    --environment "$CONTAINERAPPS_ENV" \
    --image "$IMAGE" \
    --ingress external \
    --target-port "$TARGET_PORT" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_USER" \
    --registry-password "$ACR_PASS" \
    --min-replicas 1 \
    --max-replicas 3 \
    --cpu 2 \
    --memory 4Gi >/dev/null
fi
echo "✅ Container App OK"

#################################
# 8) Application URL
#################################
APP_URL=$(az containerapp show -n "$CONTAINER_APP_NAME" -g "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')

echo ""
echo "=========================================="
echo "✅ DEPLOYMENT SUCCESSFUL"
echo "=========================================="
echo "ACR      : $ACR_NAME"
echo "Region   : $LOCATION"
echo "Resource Group: $RESOURCE_GROUP"
echo ""
echo "Application URLs:"
echo "  App URL  : https://$APP_URL"
echo "  Streamlit: https://$APP_URL"
echo ""
echo "To delete all resources:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo "=========================================="