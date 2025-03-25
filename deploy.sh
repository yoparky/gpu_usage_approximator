# File: deploy.sh
#!/bin/bash

# Set variables
SSH_HOST="your-company-ssh-host"
SSH_USER="your-ssh-username"
PROJECT_PATH="/path/to/deployment/vllm_gpu_calculator"

# Copy project files to remote server
rsync -avz --exclude 'node_modules' --exclude '__pycache__' --exclude 'venv' ./ $SSH_USER@$SSH_HOST:$PROJECT_PATH

# SSH into server and start the application
ssh $SSH_USER@$SSH_HOST << EOF
  cd $PROJECT_PATH
  docker-compose down
  docker-compose up -d
  echo "Deployment completed successfully!"
EOF