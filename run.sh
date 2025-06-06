llama stack run --image-type venv --image-name trl-post-training run.yaml

curl -s http://localhost:8321/v1/providers | jq '.data[] | select(.api=="post_training")'