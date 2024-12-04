import lightning as L
from torch import nn
import torch

from survey_map.plots import vit_plot_original_vs_reconstructed

import torch.optim as optim
from einops import rearrange, repeat
import torch.nn.functional as F

from transformer import Transformer, ViT

class Survey2DMAE(L.LightningModule):
    def __init__(
        self,
        *,
        encoder,
        decoder,
        observation_name,
        encoder_save_path = None,
        masking_ratio = 0.1,
        image_size = 32,
        learning_rate = 1e-4,
    ):
        super().__init__()
        assert masking_ratio > 0 and masking_ratio < 1, 'masking ratio must be kept between 0 and 1'

        self.should_collect_plot = True
        self.lr = learning_rate
        self.masking_ratio = masking_ratio
        self.image_size = image_size
        self.observation_name = observation_name

        # encoder parameters
        self.encoder = encoder
        self.encoder_save_path = encoder_save_path
        self.num_channels = encoder.num_channels
        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])
        self.patch_size = encoder.patch_size
        self.num_patches_side = self.image_size // self.patch_size  # Number of patches per dimension

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        # decoder parameters
        self.decoder = decoder
        self.decoder_dim = self.decoder.dim
        self.decoder_pos_emb = nn.Embedding(num_patches, self.decoder_dim)

        self.enc_to_dec = nn.Linear(encoder_dim, self.decoder_dim) if encoder_dim != self.decoder_dim else nn.Identity()
        self.mask_token = nn.Parameter(torch.randn(self.decoder_dim))

        self.to_pixels = nn.Linear(self.decoder_dim, pixel_values_per_patch)

    def forward(self, img, return_reconstructions=False):
        device = img.device

        # get patches
        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1:(num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device = device).argsort(dim = -1)
        masked_indices, unmasked_indices = rand_indices[:, :num_masked], rand_indices[:, num_masked:]

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device = device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(unmasked_indices)

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, 'd -> b n d', b = batch, n = num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(batch, num_patches, self.decoder_dim, device=device)
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss
        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)

        if return_reconstructions:
            # Reconstruct full image from patches

            reconstructed_image = torch.zeros((batch, 1, self.image_size, self.image_size))
            reconstructed_image = reconstructed_image.to(device)

            num_patches = pred_pixel_values.shape[1]

            reshaped_patches = pred_pixel_values.view(batch, num_patches, self.patch_size, self.patch_size, self.num_channels)

            for i in range(batch):
                for j in range(num_patches):
                    patch = reshaped_patches[i, j, :, :, :]
                    patch = patch.mean(dim=-1) # average across last dimension

                     # Determine patch position based on the index
                    row = (masked_indices[i, j] // self.num_patches_side) * self.patch_size
                    col = (masked_indices[i, j] % self.num_patches_side) * self.patch_size

                    # Place the reconstructed patch in the image
                    reconstructed_image[i, 0, row:row+self.patch_size, col:col+self.patch_size] = patch

            # Create a masked version of the original image
            masked_original = img.clone()
            for b in range(batch):
                for idx in masked_indices[b]:
                    row = (idx // self.num_patches_side) * self.patch_size
                    col = (idx % self.num_patches_side) * self.patch_size
                    masked_original[b, :, row:row+self.patch_size, col:col+self.patch_size] = 0

            #print(f"masked_original.device: {masked_original.device}")
            #print(f"reconstructed_image.device: {reconstructed_image.device}")
            #print(f"img.device: {img.device}")
            final_reconstruction = masked_original + reconstructed_image

            return recon_loss, final_reconstruction, masked_original

        return recon_loss

    def training_step(self, batch, batch_idx):
        inputs = batch[self.observation_name]
        # add channel dimension at index 1
        inputs = inputs.unsqueeze(1)

        recon_loss = self.forward(inputs, return_reconstructions=False)

        self.log('train_loss', recon_loss, prog_bar=True)
        return recon_loss

    def validation_step(self, batch, batch_idx):
        inputs = batch[self.observation_name]
        inputs = inputs.unsqueeze(1)

        if self.should_collect_plot:
            recon_loss, reconstructed_patches, masked_original = self.forward(inputs, return_reconstructions=True)

            fig = vit_plot_original_vs_reconstructed(inputs, masked_original, reconstructed_patches)

            self.logger.experiment["validation"].append(fig)

            self.should_collect_plot = False
        else:
            recon_loss = self.forward(inputs, return_reconstructions=False)

        self.log('val_loss', recon_loss, prog_bar=True)
        return recon_loss

    def on_train_end(self) -> None:
        if self.encoder_save_path:
            self.encoder = self.encoder.cpu()
            self.encoder.save_encoder_weights(self.encoder_save_path)

    def on_train_epoch_end(self):
        self.should_collect_plot = True

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=self.lr)

