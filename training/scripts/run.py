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
    batch_size=128,
    dropout=0.3,
    clip_value=1,
    early_stopping_patience=3,
    gamma=0.8,
    lr=2e-4,
    lr_patience=0,
    max_len=23,
    name="moviegenre",
    num_classes=10,
    num_epochs=16,
    num_steps=64,
)
