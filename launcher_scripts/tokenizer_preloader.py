from pathlib import Path
import yaml
from transformers import AutoTokenizer


def load_yaml(path):
    with open(path) as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            return {}

def load_yamls(paths):
    for path in paths:
        yield load_yaml(path)

def extract_tokenizers(conf_iter):
    tokenizers = []
    for conf in conf_iter:
        try:
            tok_conf = conf['model']['tokenizer']
            if tok_conf['library'] != 'huggingface': continue
            tokenizers.append(tok_conf['type'])
        except:
            continue
    return tokenizers


root_path = Path(__file__).parents[0] / 'conf/training/'
tokenizers = extract_tokenizers(load_yamls(map(str, root_path.rglob('*/*.yaml'))))
hardcoded_tokenizers = ['gpt2', 'bert-base-cased', 'bert-large-cased', 'bert-large-uncased']
tokenizers = list(set(hardcoded_tokenizers + tokenizers))
for tokenizer_name in tokenizers:
    print(tokenizer_name)
    AutoTokenizer.from_pretrained(tokenizer_name)
