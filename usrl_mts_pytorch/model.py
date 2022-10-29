import torch
import numpy as np
from sklearn.utils import shuffle

from usrl_mts_pytorch.losses import TripletLoss, TripletLossVaryingLength
from usrl_mts_pytorch.modules import CausalCNNEncoder

class Encoder():
    
    def __init__(self,
                 x,
                 blocks,
                 filters,
                 kernel_size,
                 encoder_length,
                 output_length):
        
        '''
        Implementation of encoder model introduced in Franceschi, J.Y., Dieuleveut, A. and Jaggi, M., 2019.
        Unsupervised scalable representation learning for multivariate time series. Advances in neural
        information processing systems, 32.
        
        Adapted from: https://github.com/White-Link/UnsupervisedScalableRepresentationLearningTimeSeries.

        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        blocks: int.
            The number of blocks of convolutional layers.
        
        filters: int.
            The number of filters (or channels) of the convolutional layers.

        kernel_size: int.
            Kernel size of the convolutional layers.

        encoder_length: int.
            Length of the encoded representation.
            
        output_length: int.
            Length of the output representations.
        '''
        
        # Shuffle the inputs.
        x = shuffle(x)
        
        # Scale the inputs.
        self.x_min = np.nanmin(x, axis=0, keepdims=True)
        self.x_max = np.nanmax(x, axis=0, keepdims=True)
        self.x = (x - self.x_min) / (self.x_max - self.x_min)
 
        # Build the model.
        self.model = CausalCNNEncoder(
            in_channels=x.shape[1],
            channels=filters,
            depth=blocks,
            reduced_size=encoder_length,
            out_channels=output_length,
            kernel_size=kernel_size
        )

        # Check if the inputs have varying length.
        self.varying = np.isnan(self.x).any()
        
        # Check if GPU is available.
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def fit(self,
            negative_samples,
            learning_rate,
            batch_size,
            epochs,
            verbose=True):
        
        '''
        Train the model.

        Parameters:
        __________________________________
        negative_samples: int.
            Number of negative examples.
            
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''

        print(f'Training on {self.device}.')
        self.model.to(self.device)
        
        # Generate the training dataset.
        dataset = torch.from_numpy(self.x).to(self.device)

        # Generate the training batches.
        batches = torch.utils.data.DataLoader(
            dataset=torch.utils.data.TensorDataset(dataset),
            batch_size=batch_size,
            shuffle=True
        )
        
        # Define the optimizer.
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        # Define the loss function.
        if self.varying:
            loss_fn = TripletLossVaryingLength(
                nb_random_samples=negative_samples,
                compared_length=None,
                negative_penalty=1,
            )
        else:
            loss_fn = TripletLoss(
                nb_random_samples=negative_samples,
                compared_length=None,
                negative_penalty=1,
            )

        # Train the model.
        self.model.train(True)
        
        for epoch in range(epochs):
            for batch in batches:
                optimizer.zero_grad()
                loss = loss_fn(
                    batch=batch[0].to(self.device),
                    encoder=self.model,
                    train=dataset,
                    save_memory=False if self.device == 'cpu' else True
                )
                loss.backward()
                optimizer.step()
            if verbose:
                print('epoch: {}, loss: {:,.6f}'.format(1 + epoch, loss))

        self.model.train(False)
    
    def predict(self, x):
    
        '''
        Generate the representations.
        
        Parameters:
        __________________________________
        x: np.array.
            Time series, array with shape (samples, channels, length) where samples is the number of time series,
            channels is the number of dimensions of each time series (1: univariate, >1: multivariate) and length
            is the length of the time series.

        Returns:
        __________________________________
        z: np.array.
            Representations, array with shape (samples, output_length) where samples is the number of time series
            and output_length is the length of the representations.
        '''
    
        # Scale the inputs.
        x = (x - self.x_min) / (self.x_max - self.x_min)
        
        # Generate the representations.
        z = self.model(torch.from_numpy(x).to(self.device))
        z = np.nan_to_num(z.detach().cpu().numpy())
    
        return z
