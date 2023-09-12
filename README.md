# Information Quality

An NLP transformer model for classifying the quality of natural language as "high" or "low"

## Installation

```bash
make develop
```

## Testing

```bash
make test
```

## Usage

```bash
.venv/bin/python scripts/train.py \
    --num-epochs 32 \
    --gamma 0.8 \
    --num-steps 16 \
    --lr 0.0001 \
    --batch-size 128 \
    --name imdbsent
```

## Training

![](tools/readme/training-screenshot.png)

## Testing

![](tools/readme/pytest-screenshot.png)
