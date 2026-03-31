import os
import torch
import models
import data_utils as utils
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

#number of target blocks
npred = 4

tokenizer = models.Tokenizer(img_size=224, patch_size=16)
collator = utils.MaskCollator(npred = npred)

data_loader_train = utils.make_imagenet(
    # dataset_name="timm/mini-imagenet", # only specify the first time to download
    local_name="mini-imagenet",
    transform=utils.make_transforms(normalization=(0, 1)), # for visuslization, no normalization
    patcher=tokenizer.encode,
    collator=collator,
    split="train"
)

context_train_data = data_loader_train


for (images, labels), enc_masks, pred_masks in data_loader_train:
    print(utils.apply_masks(images, enc_masks).shape)
    print(utils.apply_masks(images, pred_masks).shape)

lr = 1e-3 # too high
lr = 1e-4 #too low
lr = 6e-4 #too low
epochs = 10



#models
context_encoder = models.Encoder().to(device)
predictor = models.Predictor().to(device)
target_encoder = models.Encoder(positional_embeddings=context_encoder.positional_embedding).to(device)

#loss and optimizer
optimizer =  torch.optim.Adam(list({*context_encoder.parameters(), *predictor.parameters()}), lr = lr)
loss_fn = torch.nn.MSELoss()

#EMA momentum for updating the target encoder's weights
momentum = 0.996  # typical value from the paper  


loss_history = []


for epoch in range(epochs):
    logger.info(f"Epoch {epoch + 1}/{epochs}")
    pbar = tqdm(data_loader_train, desc=f"Epoch {epoch + 1}/{epochs}")
    for (images, labels), enc_masks, pred_masks in pbar:
        

        enc_masks = enc_masks.flatten(start_dim=0, end_dim=1)
        pred_masks = pred_masks.flatten(start_dim=0, end_dim=1)

        nb = device.type == "cuda"
        images = images.to(device, non_blocking=nb)
        enc_masks = enc_masks.to(device, non_blocking=nb)
        pred_masks = pred_masks.to(device, non_blocking=nb)

        optimizer.zero_grad()  # Clear previous gradients

        context = utils.apply_masks(images, enc_masks)  #[B, n_context, D]

        #get embeddings
        context_embeddings = context_encoder(context, enc_masks) #[B, n_context, n_embed]

        
        image_embeddings = target_encoder(images) #B, P, n_embed

        actual_target_embeddings = utils.apply_masks(image_embeddings, pred_masks) # [npred*B, n_target, n_embed]

        #get the prediction
        predicted_target_embeddings = predictor(context_embeddings.repeat(npred, 1, 1), enc_masks.repeat(npred, 1), pred_masks) #[npred*B, n_target, n_embed]
        
        #calculate the loss
        loss = loss_fn(predicted_target_embeddings, actual_target_embeddings)

        loss_history.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

        #backprop
        loss.backward()
        optimizer.step()

        #update the target encoder weights using EMA                                                                                                          
        with torch.no_grad():                                                                                                   
            for ctx_param, tgt_param in zip(context_encoder.parameters(), target_encoder.parameters()):                         
                tgt_param.data = momentum * tgt_param.data + (1 - momentum) * ctx_param.data 

        
        

checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)
checkpoint_path = os.path.join(checkpoint_dir, "ijepa_train.pt")
torch.save(
    {
        "context_encoder": context_encoder.state_dict(),
        "predictor": predictor.state_dict(),
        "target_encoder": target_encoder.state_dict(),
        "epochs": epochs,
    },
    checkpoint_path,
)
logger.info("Saved weights to %s", checkpoint_path)

plt.figure()
plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("MSE loss")
plt.title("I-JEPA training loss")
loss_plot_path = os.path.join(checkpoint_dir, "loss.png")
plt.savefig(loss_plot_path, dpi=150, bbox_inches="tight")
plt.close()
logger.info("Saved loss plot to %s", loss_plot_path)