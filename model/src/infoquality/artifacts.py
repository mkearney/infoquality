from transformers import AutoTokenizer, BertModel

# tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

# token map
bert_token_map = {str(k): int(v) for k, v in tokenizer.vocab.items()}  # type: ignore

# embeddings
bert_model = BertModel.from_pretrained("bert-base-cased")
bert_embeddings = bert_model.embeddings.word_embeddings.weight  # type: ignore
