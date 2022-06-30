import fire
from omegaconf import OmegaConf

from plagiarism.detector import extrinsic_plg, intrinsic_plg
from plagiarism.util import metric
from plagiarism.vectorizer import TFIDFHashing, SE


def extrinsic_se(config):
    conf = OmegaConf.load(config)

    extrinsic_plg(
        conf.extrinsic.source.pth,
        conf.extrinsic.suspicious.pth,
        conf.extrinsic.source.dir,
        conf.extrinsic.suspicious.dir,
        conf.extrinsic.index,
        conf.extrinsic.save,
        SE(),
    )


def extrinsic_tfidf(config):
    conf = OmegaConf.load(config)

    extrinsic_plg(
        conf.extrinsic.source.pth,
        conf.extrinsic.suspicious.pth,
        conf.extrinsic.source.dir,
        conf.extrinsic.suspicious.dir,
        conf.extrinsic.index,
        conf.extrinsic.save,
        TFIDFHashing(),
    )


def intrinsic(config):
    conf = OmegaConf.load(config)
    intrinsic_plg(
        conf.intrinsic.suspicious.pth,
        conf.intrinsic.suspicious.dir,
        conf.intrinsic.features,
        conf.intrinsic.save,
    )


def evaluation(config):
    conf = OmegaConf.load(config)
    print(metric(**conf.evaluation))


if __name__ == "__main__":
    fire.Fire()
