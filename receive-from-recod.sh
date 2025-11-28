rsync --progress -avz \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  lucas.nogueira@ssh.recod.ic.unicamp.br:galo/ ./
