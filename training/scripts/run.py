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

conductor(
    activation="gelu",
    batch_size=128,
    clip_value=1.0,
    dropout=0.3,
    early_stopping_patience=8,
    embedding_dimensions=200,
    gamma=0.8,
    lr=0.0005,
    lr_patience=0,
    max_len=40,
    name="moviegenre",
    num_classes=10,
    num_epochs=32,
    num_heads=8,
    num_layers=2,
    num_steps=32,
)
