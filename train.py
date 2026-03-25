import torch.nn 
import torch
import models

tokenizer = Tokenizer(img_size=224, patch_size=16)
collator = MaskCollator()

lr = 1e-3
epochs = 10

optimizer =  torch.optimizer.Adam(lr = lr)

loss_fn = torch.nn.MSELoss()

predictor = models.Predictor()
context_encoder = models.Encoder()
target_encoder = models.Encoder()




for epoch in range(epochs):
    pass