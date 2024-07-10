mkdir datasets
cd datasets

# Clone the  grade-school-math repository
git clone https://github.com/openai/grade-school-math.git

# Clone the BigBench Hard Repository
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git

# Download the CommonSenseQA Dataset
mkdir CommonSenseQA
curl -O https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl --output-dir CommonSenseQA/
