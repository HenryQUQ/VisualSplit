from ..models.CrossViT import CrossViTForPreTraining, CrossViTConfig
import datetime
from ..config import Config
from ..utils.create_folder import create_folder
import wandb
import os
from ..utils.load_dataset import load_dataset
from tqdm import tqdm
from ignite.metrics import SSIM
from accelerate import Accelerator
import accelerate
import argparse
import torch
from matplotlib import pyplot as plt

from diffusers import get_scheduler
import lpips

if int(os.environ.get("WORLD_SIZE", 1)) <= 1:
    pass
else:
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=datetime.timedelta(seconds=54000)
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Train the model")

    parser.add_argument(
        "--pretrain_model", type=str, default=None, help="pretrain model path"
    )
    parser.add_argument(
        "--section_batch", type=int, default=500, help="record loss every section_batch"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.5e-4,
        help="The learning rate for the model training",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="The batch size for the model training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="The number of epochs for the model training",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ImageNet-1k-pure",
        help="The dataset for the model training",
    )
    parser.add_argument(
        "--name", type=str, default="CrossViT", help="The name of the model"
    )

    args = parser.parse_args()
    return args


def add_parse_to_params(params, args):
    for key, value in vars(args).items():
        if value is not None:
            params[key] = value
    return params


def show_image(image, imagenet_std, imagenet_mean, title=""):
    # image is [H, W, 3]
    if len(image.shape) == 4:
        image = torch.einsum("nchw->nhwc", image).detach().cpu()[0]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis("off")
    return


def train_pipeline(params, train_loader, val_loader):

    config = CrossViTConfig()
    model = CrossViTForPreTraining(config)

    accelerator = Accelerator(
        gradient_accumulation_steps=5,
        kwargs_handlers=[
            accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
        ],
    )

    model = model.to(accelerator.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=params["learning_rate"],
        betas=(0.95, 0.999),
        weight_decay=1e-6,
        eps=1e-8,
    )

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=500 * 1,
        num_training_steps=(len(train_loader) * args.epochs),
    )

    if accelerator.is_main_process:
        start_time = datetime.datetime.now()
        train_folder, train_model_folder, train_image_logger_folder = create_folder(
            params=params,
            root_folder=Config.TRAINING_FOLDER,
            start_time=start_time,
        )
        wandb.init(
            project=f"crossvit_{Config.LOCAL}",
            name="crossvit"
            + "_"
            + params["dataset"]
            + "_"
            + start_time.strftime("%Y-%m-%dT%H-%M-%S"),
        )
        wandb.config.update(params)
        print(f"params: {params}")
        print(f"Training folder: {train_folder}")

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    if "pretrain_model" in params and params["pretrain_model"] is not None:
        unwrapped_model = accelerator.unwrap_model(model)
        if params["pretrain_model"].endswith(".safetensors"):
            from safetensors.torch import load_model

            load_model(unwrapped_model, params["pretrain_model"])
        model = accelerator.prepare_model(unwrapped_model)

        import re

        epoch_pattern = re.compile(r"epoch_(\d+)")
        init_epoch = int(epoch_pattern.search(params["pretrain_model"]).group(1)) + 1
        print(f"Loaded model from epoch {init_epoch}")

        for i in range(init_epoch * len(train_loader)):
            optimizer.zero_grad()
            optimizer.step()
            lr_scheduler.step()
    else:
        init_epoch = 0
        print("Training new model")

    lpips_loss_fn = lpips.LPIPS(net="alex").to(accelerator.device)

    for epoch in range(init_epoch, params["epochs"]):
        model.train()

        ssim_loss = SSIM(data_range=1.0)

        with tqdm(
            train_loader,
            unit="batch",
            desc=f'Epoch {epoch}/{params["epochs"]}',
            disable=accelerator.is_local_main_process,
        ) as pbar:
            for rgb_clip, edge, gray_level, source_segmented_rgb, ab in pbar:
                with accelerator.accumulate(model):
                    optimizer.zero_grad()

                    model_output = model(
                        pixel_values=rgb_clip,
                        source_edge=edge,
                        source_gray_level_histogram=gray_level,
                        source_segmented_rgb=source_segmented_rgb,
                    )

                    mse_loss = model_output["loss"]

                    lpips_loss = lpips_loss_fn(
                        model_output["logits_reshape"], rgb_clip
                    ).mean()

                    loss = 0 + 0.1 * lpips_loss

                    accelerator.backward(loss)

                    lr_scheduler.step()

                    optimizer.step()

                    one_batch_loss = {
                        "loss": loss.item(),
                        "mse_loss": mse_loss.item(),
                        "lpips_loss": lpips_loss.item(),
                    }
                    pbar.set_postfix(**one_batch_loss)
                    if accelerator.is_main_process:
                        wandb.log(one_batch_loss)

        accelerator.wait_for_everyone()

        if accelerator.is_main_process:
            model.eval()

            count = 0
            for rgb_clip, edge, gray_level, source_segmented_rgb, ab in pbar:
                try:

                    with torch.no_grad():
                        rgb_clip = rgb_clip.to(accelerator.device)
                        model_output = model(
                            pixel_values=rgb_clip,
                            source_edge=edge,
                            source_gray_level_histogram=gray_level,
                            source_segmented_rgb=source_segmented_rgb,
                        )

                        imagenet_mean = 0
                        imagenet_std = 1
                        image_save_path = os.path.join(
                            train_image_logger_folder,
                            f"epoch_{epoch}_batch_{count}.png",
                        )
                        plt.subplot(1, 2, 1)
                        show_image(
                            rgb_clip, imagenet_std, imagenet_mean, title="Source RGB"
                        )
                        plt.subplot(1, 2, 2)
                        show_image(
                            model_output["logits_reshape"],
                            imagenet_std,
                            imagenet_mean,
                            title="Reconstructed RGB",
                        )
                        plt.savefig(image_save_path)
                        accelerator.print(f"Image saved to {image_save_path}")

                except:
                    print("Error")
                count += 1
                if count >= 10:
                    break

            accelerator.save_model(
                model, os.path.join(train_model_folder, f"epoch_{epoch}")
            )

        lr_scheduler.step()

    wandb.finish()


if __name__ == "__main__":
    args = parse_args()

    params = {}

    params = add_parse_to_params(params, args)

    train_loader, val_loader, test_loader = load_dataset(
        "ImageNet-1k-pure",
        batch_size=params["batch_size"],
    )

    train_pipeline(params, train_loader, val_loader)
