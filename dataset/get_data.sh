wget https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data/python.zip
wget https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data/java.zip
wget https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data/ruby.zip
wget https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data/javascript.zip
wget https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data/go.zip
wget https://huggingface.co/datasets/code-search-net/code_search_net/resolve/main/data/php.zip

unzip python.zip
unzip java.zip
unzip ruby.zip
unzip javascript.zip
unzip go.zip
unzip php.zip
rm *.zip
rm *.pkl

python preprocess.py
rm -r */final
cd ..