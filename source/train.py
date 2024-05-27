# Import the required modules
from anomalib.data import MVTec
from anomalib.models import Patchcore
from anomalib.engine import Engine
from anomalib.loggers import AnomalibWandbLogger

def main():
  ######################################
  # Input parameters | Hyperparameters #
  ######################################
  data_root = "../dataset/MVTec"
  batch_size = 2
  image_size = 256
  num_workers= 8

  max_epochs = 10
  seed = 26

  backbone = "mobilenetv3_large_100"
  layers   = ['blocks.2.2', 'blocks.4.1', 'blocks.6.0']

  category = "bottle" #
  # backbone = "wide_resnet50_2"
  # layers = ["layer2", "layer3"]
  ######################################

  model = Patchcore(backbone, layers, pre_trained=True)
  wandb_logger = AnomalibWandbLogger(name="baseline", save_dir="results-wandb", project="MVTecAD-anomalib")
  engine = Engine(max_epochs=max_epochs, logger=wandb_logger)

  datamodule = MVTec(root=data_root,
                     category=category,
                     train_batch_size=batch_size,
                     eval_batch_size=batch_size,
                     image_size=image_size,
                     num_workers=num_workers,
                     seed=seed
                     )

  datamodule.setup()
  train_dataloader = datamodule.train_dataloader()
  val_dataloader   = datamodule.val_dataloader()
  engine.fit(model, train_dataloader, val_dataloader)

if __name__ == "__main__":
   main()