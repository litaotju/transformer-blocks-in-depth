
# Skip: 13B 175B, not enough memory

for config in small medium large xl 2.7B 6.7B; do 
    echo "Testing $config"
    python src/gpt.py $config
done