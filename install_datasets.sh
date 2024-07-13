mkdir datasets
cd datasets

# Clone the  grade-school-math repository
git clone https://github.com/openai/grade-school-math.git

# Clone the BigBench Hard Repository
git clone https://github.com/suzgunmirac/BIG-Bench-Hard.git

# Download the CommonSenseQA Dataset
mkdir CommonSenseQA
curl -O https://s3.amazonaws.com/commensenseqa/train_rand_split.jsonl --output-dir CommonSenseQA/

# Download the HotpotQA Wiki Dump
mkdir HotpotQA

curl -O https://nlp.stanford.edu/projects/hotpotqa/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 --output-dir HotpotQA/
# verify that we have the whole thing
unameOut="$(uname -s)"
case "${unameOut}" in
    Darwin*)    MD5SUM="md5 -r";;
    *)          MD5SUM=md5sum
esac
if [ `$MD5SUM HotpotQA/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2 | awk '{print $1}'` == "01edf64cd120ecc03a2745352779514c" ]; then
    echo "Downloaded the processed Wikipedia dump from the HotpotQA website. Everything's looking good, so let's extract it!"
else
    echo "The md5 doesn't seem to match what we expected, try again?"
    exit 1
fi
cd HotpotQA
tar -xjvf enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
# clean up
rm enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2
cd ..

# Download the HotpotQA Test and Dev sets
curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_fullwiki_v1.json --output-dir HotpotQA/
curl -O http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_test_fullwiki_v1.json --output-dir HotpotQA/
