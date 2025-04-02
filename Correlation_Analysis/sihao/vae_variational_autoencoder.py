import lightning.pytorch as pl
import torch 
import torch.nn.functional as F
from torch import optim 
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import random 

class variational_autoencoder(pl.LightningModule):
    def __init__(self, encoder, decoder, lr, kld_weight, unique_stims):
        super().__init__()
        self.lr = lr # learning rate 
        self.encoder = encoder
        self.decoder = decoder
        self.kld_weight = kld_weight # KL-Divergence weight 
        self.unique_stims = unique_stims # titles for tSNE 
        self.validation_mus = [] # values for tSNE 
        self.validation_labels = [] # colors for tSNE 
        self.test_mus = [] # values for tSNE 
        self.test_labels = [] # colors for tSNE 
        self.validation_x = [] # for reconstruction plots 
        self.validation_x_hat = [] # for reconstruction plots 
        

    def training_step(self, batch, batch_idx):
        """
        Training step in VAE 
        Returns loss used to update parameter weights of all layers 
        """
        # Retrieve data batch 
        x, y = batch  
        # Encode 
        mu, logvar = self.encoder(x)
        # Draw sample 
        sample = self.reparameterize(mu, logvar)
        # Decode sample 
        x_hat = self.decoder(sample) 
        # Calculate loss 
        loss_dict = self.loss_function(x, x_hat, logvar, mu)
        self.log_dict(loss_dict, logger=True, on_step=False, on_epoch=True, prog_bar=False)
        # Return loss 
        return loss_dict['loss']


    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian 
        :param logvar: (Tensor) Standard deviation of the latent Gaussian 
        :return: (Tensor) [B x D]
        """
        # Calculate standard deviation 
        std = torch.exp(0.5 * logvar)
        # Draw random sample 
        random_sample = mu + std * torch.randn_like(std)
        return random_sample
    

    def loss_function(self, x: torch.Tensor, x_hat: torch.Tensor, logvar: torch.Tensor, mu: torch.Tensor) -> dict:
        """
        Computes the VAE loss function.
        :param x: Input
        :param x_hat: Reconstruction 
        :param logvar: logvar of latent dimensions 
        :param mu: mu of latent dimensions 
        """
        # Reconstruction loss (MSE)
        recon_loss = torch.mean((x - x_hat)**2 * (x**2)) 
        # KL Divergence Loss 
        kld_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        # Adjust using kld_weight 
        kld_loss_weighted = self.kld_weight * kld_loss 
        # Loss = MSE + KLD 
        loss = recon_loss + kld_loss_weighted 
        output = {'loss': loss, 'reconstruction_loss': recon_loss, 'kld_loss': kld_loss, 'kld_loss_weighted': kld_loss_weighted}
        return output
    
    def configure_optimizers(self):
        """
        Stochastic Gradient Descent with learning rate lr
        """
        optimizer = optim.Adam(self.parameters(), lr=self.lr) 
        return optimizer
    
    def generate_mu(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input x, returns the latent mu after passing through the encoder 
        :param x: (Tensor)
        :return mu: (Tensor) 
        """
        # Encode 
        mu, _ = self.encoder(x)
        return mu
    
    def generate_xhat(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input x, returns the reconstructed x_hat from the latent variables
        :param x: (Tensor)
        :return mu: (Tensor) 
        """
        # Encode 
        mu, logvar = self.encoder(x)
        # Draw sample 
        sample = self.reparameterize(mu, logvar)
        # Decode sample 
        x_hat = self.decoder(sample) 
        return x_hat
            
    def validation_step(self, batch, batch_idx):
        """
        Validation step in the VAE. 
        Validation frequency is set in the Trainer instance 
        Validation logs the loss (with KL Divergence) and reconstruction loss (MSE)
        of a held out dataset (validation set)
        """
        # Retrieve data batch 
        x, y = batch  
        # Encode 
        mu, logvar = self.encoder(x)
        # Save labels and reparameterized latents for tSNE 
        self.validation_labels.extend(y.cpu().numpy())
        self.validation_mus.extend(mu.cpu().numpy())
        # Draw sample 
        sample = self.reparameterize(mu, logvar)
        # Decode sample 
        x_hat = self.decoder(sample) 
        # Calculate loss 
        loss_dict = self.loss_function(x, x_hat, logvar, mu)
        # Log the loss 
        output = {'val_loss': loss_dict["loss"], 'val_reconstruction_loss': loss_dict["reconstruction_loss"]}
        self.log_dict(output, logger=True, on_step=False, on_epoch=True, prog_bar=False)
        # Store x and x_hat for reconstruction plots 
        if batch_idx == 0: 
            self.validation_x = x.cpu().numpy() 
            self.validation_x_hat = x_hat.cpu().numpy() 

    def test_step(self, batch, batch_idx):
        """
        Test step in the VAE: Makes tSNE for ALL data 
        """
        # Retrieve data batch 
        x, y = batch  
        # Encode 
        mu, _ = self.encoder(x)
        # Save labels and reparameterized latents for tSNE 
        self.validation_labels.extend(y.cpu().numpy())
        self.validation_mus.extend(mu.cpu().numpy())
        
    def on_test_epoch_end(self):
        """
        After the last batch of testing, create the tSNE 
        """
        # Stack latents from all batches 
        validation_mus = np.stack(self.validation_mus)
        validation_labels = np.stack(self.validation_labels)
        # Generate plot and save to Tensorboard 
        plot = self.make_tsne_figure(validation_mus, validation_labels)
        self.logger.experiment.add_figure("ALL DATA LATENT SPACE tSNE", plot, global_step=self.current_epoch+1)
    

    def on_validation_epoch_end(self):
        """
        After the last batch of validation for a given epoch: 
        Creates a tSNE of the latent space for all batches
        Saves the image to the Tensorboard 
        """
        if ((self.current_epoch+1) % 2) == 0: 
            plot = self.make_reconstruction_plots(self.validation_x, self.validation_x_hat)
            self.logger.experiment.add_figure("Reconstruction Plot", plot, global_step=self.current_epoch+1)
            
        if ((self.current_epoch + 1) % 10) == 0:
            # Stack latents from all batches 
            validation_mus = np.stack(self.validation_mus)
            validation_labels = np.stack(self.validation_labels)
            
            # Generate plot and save to Tensorboard 
            plot = self.make_tsne_figure(validation_mus, validation_labels)
            self.logger.experiment.add_figure("Latent Space tSNE", plot, global_step=self.current_epoch+1)
        
        # Reset validation_mus 
        self.validation_mus = []
        self.validation_labels = []

    def make_tsne_figure(self, input, labels): 
        """
        Returns a fig containing a 2D tSNE for a given input
        :param input: Neuron activity in latent space (np.ndarray)
        :param labels: Labels for each neuron (np.ndarray) 
        :return fig: (plt.figure) 
        """
        tsne_model = TSNE(n_components=2, n_iter=1000, learning_rate='auto', init='pca', perplexity=30)
        transformed_latents = tsne_model.fit_transform(input)
        fig, axs = plt.subplots(13, 2, figsize=(8, 30))
        axs = axs.ravel()
        for i in range(26):
            axs[i].scatter(transformed_latents[:,0], transformed_latents[:,1], c=labels[:,i], s=0.1)
            axs[i].set_title(self.unique_stims[i])
            axs[i].axis(False)
        return fig 
    

    def make_reconstruction_plots(self, x, x_hat):
        """
        Returns a fig containing a plot of all the neurons in the batch 
        :param x: Activity Traces (np.ndarray) [batch, 1, time]  
        :param x_hat: Reconstructions (np.ndarray) [batch, 1, time]
        :return fig: (plt.figure) 
        """
        n = x.shape[0]
        # Reshape 
        x = np.reshape(x, (x.shape[0], x.shape[1]*x.shape[2]))
        x_hat = np.reshape(x_hat, (x_hat.shape[0], x_hat.shape[1]*x_hat.shape[2]))
        # Plot 50 random neurons 
        fig, ax = plt.subplots(n, 1, figsize=(8,15))
        for i in range(n):
            ax[i].plot(x[i,:],linewidth=0.5,c='blue')
            ax[i].plot(x_hat[i,:],linewidth=0.5,c='red')
            ax[i].axis(False)
        ax[0].legend(["x","x_hat"])
        return fig

    