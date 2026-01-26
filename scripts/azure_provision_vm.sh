#!/bin/bash
# =============================================================================
# D4RT Azure VM Provisioning Script
# =============================================================================
#
# This script creates Azure resources for D4RT training using Azure CLI.
#
# Prerequisites:
#   - Azure CLI installed: curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash
#   - Logged in: az login
#   - Subscription set: az account set --subscription "YOUR_SUBSCRIPTION"
#
# Usage:
#   ./scripts/azure_provision_vm.sh [vm_size] [num_nodes]
#
# Examples:
#   # Single A100 80GB
#   ./scripts/azure_provision_vm.sh Standard_NC24ads_A100_v4 1
#
#   # 8x A100 40GB
#   ./scripts/azure_provision_vm.sh Standard_ND96asr_v4 1
#
#   # Multi-node: 4 nodes x 8 GPUs = 32 GPUs
#   ./scripts/azure_provision_vm.sh Standard_ND96asr_v4 4
# =============================================================================

set -e

# =============================================================================
# Configuration
# =============================================================================
VM_SIZE="${1:-Standard_NC24ads_A100_v4}"
NUM_NODES="${2:-1}"
RESOURCE_GROUP="${RESOURCE_GROUP:-d4rt-training-rg}"
LOCATION="${LOCATION:-eastus}"  # Check GPU availability in your region
VM_NAME_PREFIX="${VM_NAME_PREFIX:-d4rt-node}"
ADMIN_USERNAME="${ADMIN_USERNAME:-azureuser}"
SSH_KEY_PATH="${SSH_KEY_PATH:-~/.ssh/id_rsa.pub}"

# Storage account for data
STORAGE_ACCOUNT="${STORAGE_ACCOUNT:-d4rtstorage$RANDOM}"

echo "=============================================="
echo "D4RT Azure VM Provisioning"
echo "=============================================="
echo "Resource Group: $RESOURCE_GROUP"
echo "Location: $LOCATION"
echo "VM Size: $VM_SIZE"
echo "Number of Nodes: $NUM_NODES"
echo "=============================================="
echo ""

# Check if logged in
if ! az account show &> /dev/null; then
    echo "ERROR: Not logged in to Azure. Run: az login"
    exit 1
fi

# =============================================================================
# Create Resource Group
# =============================================================================
echo "[1/6] Creating resource group..."
az group create \
    --name "$RESOURCE_GROUP" \
    --location "$LOCATION" \
    --output table

# =============================================================================
# Create Virtual Network (for multi-node communication)
# =============================================================================
echo "[2/6] Creating virtual network..."
VNET_NAME="${RESOURCE_GROUP}-vnet"
SUBNET_NAME="${RESOURCE_GROUP}-subnet"

az network vnet create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$VNET_NAME" \
    --address-prefix 10.0.0.0/16 \
    --subnet-name "$SUBNET_NAME" \
    --subnet-prefix 10.0.0.0/24 \
    --output table

# =============================================================================
# Create Network Security Group
# =============================================================================
echo "[3/6] Creating network security group..."
NSG_NAME="${RESOURCE_GROUP}-nsg"

az network nsg create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$NSG_NAME" \
    --output table

# Allow SSH
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name AllowSSH \
    --priority 1000 \
    --destination-port-ranges 22 \
    --access Allow \
    --protocol Tcp \
    --output table

# Allow NCCL communication between nodes (port 29500-29510)
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name AllowNCCL \
    --priority 1001 \
    --destination-port-ranges 29500-29510 \
    --access Allow \
    --protocol Tcp \
    --source-address-prefixes 10.0.0.0/24 \
    --output table

# Allow TensorBoard (port 6006)
az network nsg rule create \
    --resource-group "$RESOURCE_GROUP" \
    --nsg-name "$NSG_NAME" \
    --name AllowTensorBoard \
    --priority 1002 \
    --destination-port-ranges 6006 \
    --access Allow \
    --protocol Tcp \
    --output table

# =============================================================================
# Create Storage Account for Data
# =============================================================================
echo "[4/6] Creating storage account..."
# Make storage account name valid (lowercase, no special chars)
STORAGE_ACCOUNT=$(echo "$STORAGE_ACCOUNT" | tr '[:upper:]' '[:lower:]' | tr -cd 'a-z0-9' | head -c 24)

az storage account create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$STORAGE_ACCOUNT" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --output table

# Create container for training data
az storage container create \
    --account-name "$STORAGE_ACCOUNT" \
    --name "d4rt-training-data" \
    --output table

# =============================================================================
# Create VMs
# =============================================================================
echo "[5/6] Creating $NUM_NODES VM(s)..."

# Cloud-init script for automatic setup
CLOUD_INIT=$(cat <<'EOF'
#cloud-config
package_update: true
package_upgrade: true

packages:
  - build-essential
  - git
  - wget
  - curl
  - htop
  - tmux

runcmd:
  # Download setup script
  - wget -O /home/azureuser/setup.sh https://raw.githubusercontent.com/jiangyurong609/d4rt-pytorch/main/scripts/setup_azure_vm.sh || echo "Script download skipped"
  - chmod +x /home/azureuser/setup.sh || true
  - chown azureuser:azureuser /home/azureuser/setup.sh || true
EOF
)

# Create temp file for cloud-init
CLOUD_INIT_FILE=$(mktemp)
echo "$CLOUD_INIT" > "$CLOUD_INIT_FILE"

declare -a VM_IPS

for i in $(seq 0 $((NUM_NODES - 1))); do
    VM_NAME="${VM_NAME_PREFIX}-${i}"
    echo "Creating VM: $VM_NAME"

    # Create public IP
    az network public-ip create \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-ip" \
        --sku Standard \
        --allocation-method Static \
        --output table

    # Create NIC
    az network nic create \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-nic" \
        --vnet-name "$VNET_NAME" \
        --subnet "$SUBNET_NAME" \
        --network-security-group "$NSG_NAME" \
        --public-ip-address "${VM_NAME}-ip" \
        --accelerated-networking true \
        --output table

    # Create VM
    az vm create \
        --resource-group "$RESOURCE_GROUP" \
        --name "$VM_NAME" \
        --nics "${VM_NAME}-nic" \
        --image "Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest" \
        --size "$VM_SIZE" \
        --admin-username "$ADMIN_USERNAME" \
        --ssh-key-values "$SSH_KEY_PATH" \
        --os-disk-size-gb 256 \
        --custom-data "$CLOUD_INIT_FILE" \
        --output table

    # Get IP addresses
    PUBLIC_IP=$(az network public-ip show \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-ip" \
        --query ipAddress -o tsv)

    PRIVATE_IP=$(az network nic show \
        --resource-group "$RESOURCE_GROUP" \
        --name "${VM_NAME}-nic" \
        --query ipConfigurations[0].privateIPAddress -o tsv)

    VM_IPS+=("$VM_NAME: Public=$PUBLIC_IP, Private=$PRIVATE_IP")

    if [ $i -eq 0 ]; then
        MASTER_PRIVATE_IP="$PRIVATE_IP"
        MASTER_PUBLIC_IP="$PUBLIC_IP"
    fi
done

rm "$CLOUD_INIT_FILE"

# =============================================================================
# Output Summary
# =============================================================================
echo ""
echo "=============================================="
echo "Provisioning Complete!"
echo "=============================================="
echo ""
echo "VMs Created:"
for vm_info in "${VM_IPS[@]}"; do
    echo "  $vm_info"
done
echo ""
echo "Storage Account: $STORAGE_ACCOUNT"
echo ""
echo "=============================================="
echo "Next Steps"
echo "=============================================="
echo ""
echo "1. SSH into the master node:"
echo "   ssh $ADMIN_USERNAME@$MASTER_PUBLIC_IP"
echo ""
echo "2. Run the setup script on each node:"
echo "   ./setup.sh"
echo ""
echo "3. Get storage account key:"
echo "   az storage account keys list --account-name $STORAGE_ACCOUNT --query '[0].value' -o tsv"
echo ""
echo "4. Upload training data:"
echo "   export AZURE_STORAGE_ACCOUNT=$STORAGE_ACCOUNT"
echo "   export AZURE_STORAGE_KEY=<key_from_step_3>"
echo "   ./scripts/prepare_data_azure.py --upload-only"
echo ""
echo "5. Download data on VMs:"
echo "   ./scripts/download_from_azure.sh kubric ~/d4rt_data"
echo ""

if [ $NUM_NODES -gt 1 ]; then
    echo "6. Start multi-node training:"
    echo ""
    echo "   # On master node (${VM_NAME_PREFIX}-0):"
    echo "   MASTER_ADDR=$MASTER_PRIVATE_IP NODE_RANK=0 NNODES=$NUM_NODES ./scripts/azure_multinode_launch.sh"
    echo ""
    echo "   # On each worker node (${VM_NAME_PREFIX}-1, ${VM_NAME_PREFIX}-2, ...):"
    echo "   MASTER_ADDR=$MASTER_PRIVATE_IP NODE_RANK=<node_index> NNODES=$NUM_NODES ./scripts/azure_multinode_launch.sh"
else
    echo "6. Start training:"
    echo "   ./scripts/train_azure.sh multi  # for all GPUs on the VM"
fi

echo ""
echo "=============================================="
echo "Cost Estimate"
echo "=============================================="
case $VM_SIZE in
    "Standard_NC24ads_A100_v4")
        HOURLY_COST="3.67"
        GPU_INFO="1x A100 80GB"
        ;;
    "Standard_ND96asr_v4")
        HOURLY_COST="27.20"
        GPU_INFO="8x A100 40GB"
        ;;
    "Standard_ND96amsr_A100_v4")
        HOURLY_COST="32.77"
        GPU_INFO="8x A100 80GB"
        ;;
    *)
        HOURLY_COST="unknown"
        GPU_INFO="varies"
        ;;
esac

TOTAL_HOURLY=$(echo "$HOURLY_COST * $NUM_NODES" | bc 2>/dev/null || echo "unknown")
echo "VM Size: $VM_SIZE ($GPU_INFO)"
echo "Hourly cost per VM: \$$HOURLY_COST"
echo "Total hourly cost ($NUM_NODES VMs): \$$TOTAL_HOURLY"
echo ""
echo "IMPORTANT: Remember to deallocate VMs when not training!"
echo "  az vm deallocate --resource-group $RESOURCE_GROUP --name ${VM_NAME_PREFIX}-0"
echo ""
echo "To delete all resources when done:"
echo "  az group delete --name $RESOURCE_GROUP --yes --no-wait"
echo "=============================================="
