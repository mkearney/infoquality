from subprocess import Popen

from infoquality.hyperparameters import HyperParameters


class Conductor:
    cmd: str = "training/.venv/bin/python training/scripts/train.py"
    defaults: HyperParameters = HyperParameters()

    def as_arg(self, k, v):
        k = k.replace("_", "-")
        return f"--{k} {v}"

    def run(self, command):
        p = Popen(command, shell=True)
        p.communicate()

    def args(self, **kwargs):
        return [
            self.as_arg(k, v) for k, v in HyperParameters(**kwargs).__dict__.items()
        ]

    def __call__(self, *args, **kwargs):
        args = self.args(**kwargs)
        call = self.cmd + " \\\n    " + " \\\n    ".join(args)
        self.run(call)


conductor = Conductor()

# | pretrained_model                          |   parameters |   size_mb |
# |-------------------------------------------|-------------:|----------:|
# | albert-base-v2                            |    11,692,29 |      44.6 |
# | squeezebert/squeezebert-mnli-headless     |   51,102,474 |     194.9 |
# | distilbert-base-uncased                   |   66,961,674 |     255.4 |
# | distilbert-base-uncased-distilled-squad   |   66,961,674 |     255.4 |
# | distilroberta-base                        |   82,127,118 |     313.3 |
# | distilgpt2                                |   82,118,922 |     313.3 |
# | bert-base-uncased                         |  109,490,954 |     417.7 |
# | funnel-transformer/small-base             |  116,209,930 |     443.3 |
# | roberta-base                              |  124,654,350 |     475.5 |
# | roberta-base-openai-detector              |  124,654,350 |     475.5 |
# | distilbert-base-multilingual-cased        |  135,332,874 |     516.3 |
# | microsoft/deberta-base                    |  139,200,522 |     531.0 |
# | xlm-roberta-base                          |  278,052,366 |   1,060.7 |
# | microsoft/deberta-v2-xlarge               |  886,969,866 |   3,383.5 |
conductor(
    batch_size=128,
    dropout=0.1,
    clip_value=0,
    early_stopping_patience=4,
    fraction=0.2,
    gamma=0.8,
    lr=1e-4,
    lr_patience=0,
    max_len=32,
    model="distilbert-base-multilingual-cased",
    name="moviegenre",
    num_classes=10,
    num_epochs=32,
    num_steps=16,
)
