import pytest
from composer.trainer.trainer_hparams import TrainerHparams

from tests.test_hparams import walk_model_yamls


@pytest.mark.timeout(40)
@pytest.mark.parametrize("hparams_file", walk_model_yamls())
def test_model_eval(hparams_file: str):

    if "timm" in hparams_file:
        pytest.importorskip("timm")
    if "vit" in hparams_file:
        pytest.importorskip("vit_pytorch")
    if hparams_file in ["unet.yaml"]:
        pytest.importorskip("monai")
    if "deeplabv3" in hparams_file:
        pytest.importorskip("mmseg")
    hparams = TrainerHparams.create(hparams_file, cli_args=False)
    assert isinstance(hparams, TrainerHparams)

    hparams.max_duration = "1ba"
    if hparams.evaluators is not None:
        for evaluator in hparams.evaluators:
            evaluator.eval_dataset.
    hparams.evaluators



    
