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

    def __call__(self, fraction: float = 1.0, *args, **kwargs):
        args = self.args(**kwargs)
        args += ["--fraction", str(fraction)]
        call = self.cmd + " \\\n    " + " \\\n    ".join(args)
        self.run(call)


conductor = Conductor()

conductor(
    batch_size=256,
    best_metric="acc",
    dropout=0.5,
    lr=3e-4,
    early_stopping_patience=5,
    max_len=60,
    model="distilbert-base-uncased",
    name="moviegenre",
    num_classes=10,
    num_epochs=62,
    num_steps=4,
)
