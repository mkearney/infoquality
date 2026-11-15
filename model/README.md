# infoquality

Fine-tune pretrained transformers to classify text (e.g. the quality of
natural language as "high" or "low"), and load the trained models for
prediction.

```python
from infoquality import Predictor

predictor = Predictor("path/to/saved/model")
predictor.predict(["Some text to classify", "another"])
# [Prediction(label='high', proba=[0.88, 0.12]), ...]
```

See the [repository](https://github.com/mkearney/infoquality) for the
training scripts and documentation.
