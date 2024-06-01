# Import the required modules
from anomalib.data import MVTec
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.loggers import AnomalibWandbLogger


import torch
from lightning.pytorch.callbacks import ModelCheckpoint

from pathlib import Path
from uuid import uuid4
from tqdm import tqdm

import click

def train_category(hparams, category):
    model = Patchcore(hparams["backbone"],
                      hparams["layers"],
                      pre_trained=True,
                      num_neighbors=hparams["num_neighbors"])

    wandb_logger = AnomalibWandbLogger(name=hparams["exp_name"],
                                       save_dir=hparams["exp_dir"],
                                       project="MVTecAD-anomalib")

    # #######################
    # # Callbacks functions #
    # #######################
    callbacks = []
    # checkpoint_callback = ModelCheckpoint(dirpath=exp_dir, filename='{epoch}-{val_loss:.2f}.pth')
    # callbacks.append(checkpoint_callback)

    ##################
    # Trainer Module #
    ##################
    engine = Engine(callbacks=callbacks,
                    max_epochs=hparams["max_epochs"],
                    logger=wandb_logger,
                    default_root_dir=hparams["exp_dir"])

    ##############
    # Datamodule #
    ##############
    datamodule = MVTec(root=hparams["data_root"],
                        category=category,
                        train_batch_size=hparams["batch_size"],
                        eval_batch_size=hparams["batch_size"],
                        image_size=hparams["image_size"],
                        num_workers=hparams["num_workers"],
                        seed=hparams["seed"]
                        )

    datamodule.setup()
    train_dataloader = datamodule.train_dataloader()
    val_dataloader   = datamodule.val_dataloader()
    engine.fit(model, train_dataloader, val_dataloader)

    #######################################
    # save state dict for inference later #
    #######################################
    print("[INFO] Saving latest state dict for inference category {} ...".format(category))
    state_dict = model.model.state_dict()
    trained_weights_filename = Path(hparams["exp_dir"]) / Path(category + ".pth")
    torch.save(state_dict, trained_weights_filename)

    del model
    del datamodule
    del engine

@click.command()
@click.option("--category", required=True)
@click.option("--exp_id", default=None)
def main(category, exp_id=None):
    ######################################
    # Input parameters | Hyperparameters #
    ######################################
    hparams = {}

    hparams["data_root"] = "dataset/MVTec"
    hparams["category"] = category

    hparams["batch_size"] = 2
    hparams["image_size"] = 256
    hparams["num_workers"]= 8

    hparams["max_epochs"] = 10
    hparams["seed"] = 26
    hparams["num_neighbors"] = 9

    hparams["model_type"] = "Patchcore"
    hparams["backbone"] = "mobilenetv3_large_100"
    hparams["layers"]   = ['blocks.2.2', 'blocks.4.1', 'blocks.6.0']
    ######################################

    torch.set_float32_matmul_precision('medium')

    if exp_id is None:
        exp_id = uuid4().hex[:4]
    exp_name= "{}-{}-{}".format(hparams["model_type"], hparams["backbone"], exp_id)
    exp_dir = Path("experiments") / Path(exp_name)
    hparams["exp_dir"] = exp_dir
    hparams["exp_name"] = exp_name

    print("[INFO] Training for category: {}".format(category))
    train_category(hparams, category)

if __name__ == "__main__":
   main()