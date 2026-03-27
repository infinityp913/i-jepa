import torch
import models
import data_utils as utils
import matplotlib.pyplot as plt

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

lr = 1e-3
epochs = 10



#models
context_encoder = models.Encoder()
predictor = models.Predictor(context_encoder.positional_embedding)
target_encoder = models.Encoder(positional_embeddings=context_encoder.positional_embedding)

#loss and optimizer
optimizer =  torch.optim.Adam(list({*context_encoder.parameters(), *predictor.parameters()}), lr = lr)
loss_fn = torch.nn.MSELoss()

#EMA momentum for updating the target encoder's weights
momentum = 0.996  # typical value from the paper  


loss_history = []


for epoch in range(epochs):
    print("epoch: ", epoch, "/", epochs, sep="")
    for (images, labels), enc_masks, pred_masks in data_loader_train:
        

        enc_masks = enc_masks.flatten(start_dim=0, end_dim=1)
        pred_masks = pred_masks.flatten(start_dim=0, end_dim=1)

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

        #backprop
        loss.backward()
        optimizer.step()

        #update the target encoder weights using EMA                                                                                                          
        with torch.no_grad():                                                                                                   
            for ctx_param, tgt_param in zip(context_encoder.parameters(), target_encoder.parameters()):                         
                tgt_param.data = momentum * tgt_param.data + (1 - momentum) * ctx_param.data 

        
        

plt.plot(loss_history)