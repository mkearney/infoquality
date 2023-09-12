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
    batch_size=64,
    dropout=0.35,
    lr=2e-4,
    max_len=50,
    model="distilbert-base-uncased",
    name="moviegenre",
    num_classes=10,
    num_epochs=62,
    num_steps=16,
)
