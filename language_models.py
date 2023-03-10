import torch
import torch.nn.functional as F

class BigramLanguageModel():
    """
    A class representing a bigram language model trained with maximum likelihood estimation.

    Attributes:
    - x_data (torch.Tensor): a 1xM tensor of integer inputs
    - y_data (torch.Tensor): a 1xM tensor of integer outputs
    - num_classes (int): the number of possible tokens in the input data
    - weights (torch.Tensor): a num_classes x num_classes tensor of neuron weights
    - decode (dict): a dictionary for translating tokens to integers

    Methods:
    - train(epochs, learn_rate, verbose): trains the model on the input data
    - generate(num, generator, verbose): generates num sequences of tokens from the model
    """
    def __init__(self,x_data:torch.Tensor,y_data:torch.Tensor,num_classes=None,decode=None):
        """
        Initializes a new instance of the BigramLanguageModel class.

        Args:
        - x_data (torch.Tensor): a 1xM tensor of integer inputs
        - y_data (torch.Tensor): a 1xM tensor of integer outputs
        - num_classes (int, optional): the number of possible tokens in the input data
        - decode (dict, optional): a dictionary for translating tokens to integers
        """
        self.x_data = x_data # a 1xB array of integer inputs
        self.y_data = y_data # a 1xB array of integer outputs
        self.x_batch = self.x_data
        self.y_batch = self.y_data

        # Determine the number of possible tokens from the input data
        if num_classes is None:
            num_classes = max([x.item() for x in self.x_data]) + 1
        self.num_classes = num_classes

        self.num_classes = num_classes if num_classes else max([x.item() for x in self.x_data])+1

        # Initialize the neuron weights
        self.weights = torch.zeros((self.num_classes, self.num_classes), requires_grad=True)

        # Set the dictionary for token decoding
        self.decode = decode

    def train(self, epochs=100, learn_rate=1e-3, batch_size=None, verbose=False, generator=None):
        """
        Trains the model on the input data.

        Args:
        - epochs (int, optional): the number of training epochs
        - learn_rate (float, optional): the learning rate for the optimizer
        - verbose (bool, optional): whether to print training progress to stdout

        Returns:
        - weights (torch.Tensor): the trained neuron weights
        - loss (float): the final loss value
        """
        for i in range(epochs):

            # Forward pass
            if batch_size is not None:
                # selecting `batch_size` random samples from the x_data
                idxs = torch.multinomial(torch.zeros(len(self.x_data))+1, batch_size, replacement=True, generator=generator)
                self.x_batch = self.x_data[idxs.tolist()]
                self.y_batch = self.y_data[idxs.tolist()]
            else:
                self.x_batch = self.x_data
                self.y_batch = self.y_data
            
            x_enc = F.one_hot(self.x_batch, num_classes=self.num_classes).float() # one hot encoding
            logits = x_enc @ self.weights # x•W for every neuron simultaneously
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)
            loss = -probs[torch.arange(self.x_batch.nelement()), self.y_batch].log().mean() # + (W**2).mean()
            if verbose: print(f"Epoch {i}. Loss: {loss.item():.4f}")

            # Backward pass
            self.weights.grad = None # set the gradient to zero, but more performantly than = 0
            loss.backward()

            # Update the weights
            assert(self.weights.grad is not None)
            self.weights.data += -learn_rate * self.weights.grad
        
        return self.weights, loss.item()

    def generate(self, num:int=10, generator:torch.Generator|None=None, max_length=100, verbose=False):
        """
        Generates `num` sequences of tokens using the trained bigram language model.

        Args:
            num (int): The number of sequences to generate. Defaults to 10.
            generator (torch.Generator, optional): Generator used for random number generation. Defaults to None.
            verbose (bool, optional): If True, prints each generated sequence. Defaults to False.

        Returns:
            list: A list of length `num`, where each element is a list of integers representing the generated sequence.
        """
        # generated = []

        for _ in range(num):
            out = []
            idx = 0
            n = 0
            while n < max_length:
                n += 1
                x_enc = F.one_hot(torch.tensor([idx]), num_classes=self.num_classes).float()
                logits = x_enc @ self.weights # log counts
                counts = logits.exp()
                probs = counts / counts.sum(1, keepdims=True)
                idx = torch.multinomial(probs, 1, replacement=True, generator=generator).item()
                if self.decode is None:
                    out.append(idx) 
                else:
                    out.append(self.decode[idx])
            # generated.append(out)
            if verbose: 
                print(''.join(out))
        return

class TrigramLanguageModel():
    """
    A class representing a trigram language model trained with maximum likelihood estimation.

    Attributes:
    - x_data (torch.Tensor): a 2xM tensor of integer inputs
    - y_data (torch.Tensor): a 1xM tensor of integer outputs
    - num_classes (int): the number of possible tokens in the input data
    - weights (torch.Tensor): a num_classes x num_classes tensor of neuron weights
    - decode (dict): a dictionary for translating tokens to integers

    Methods:
    - train(epochs, learn_rate, verbose): trains the model on the input data
    - generate(num, generator, verbose): generates num sequences of tokens from the model
    """
    def __init__(self,x_data:torch.Tensor,y_data:torch.Tensor,num_classes=None,decode=None):
        """
        Initializes a new instance of the BigramLanguageModel class.

        Args:
        - x_data (torch.Tensor): a 2xM tensor of integer inputs
        - y_data (torch.Tensor): a 1xM tensor of integer outputs
        - num_classes (int, optional): the number of possible tokens in the input data
        - decode (dict, optional): a dictionary for translating tokens to integers
        """
        self.x_data = x_data # a 2xM array of integer inputs
        self.y_data = y_data # a 1xM array of integer outputs
        self.x_batch = self.x_data
        self.y_batch = self.y_data

        # Determine the number of possible tokens from the input data
        if num_classes is None:
            num_classes = max([x.item() for x in self.x_data]) + 1
        self.num_classes = num_classes

        self.num_classes = num_classes if num_classes else max([x.item() for x in self.x_data])+1

        # Initialize the neuron weights
        self.weights = torch.zeros((self.x_data.shape[1]*num_classes, num_classes), requires_grad=True)

        # Set the dictionary for token decoding
        self.decode = decode

    def train(self, epochs=100, learn_rate=1e-3, batch_size=None, verbose=False, generator=None):
        """
        Trains the model on the input data.

        Args:
        - epochs (int, optional): the number of training epochs
        - learn_rate (float, optional): the learning rate for the optimizer
        - verbose (bool, optional): whether to print training progress to stdout

        Returns:
        - weights (torch.Tensor): the trained neuron weights
        - loss (float): the final loss value
        """
        for i in range(epochs):

            # Forward pass
            if batch_size is not None:
                # selecting `batch_size` random samples from the x_data
                idxs = torch.multinomial(torch.zeros(len(self.x_data))+1, batch_size, replacement=True, generator=generator)
                self.x_batch = self.x_data[idxs.tolist()] # M x 2 array
                self.y_batch = self.y_data[idxs.tolist()] # M x 1 array
            else:
                self.x_batch = self.x_data # M x 2 array
                self.y_batch = self.y_data # M x 1 array
            
            # forward pass
            x_enc = F.one_hot(self.x_batch, num_classes=self.num_classes).float()
            x_enc_flattened = x_enc.reshape(batch_size,-1)
            logits = x_enc_flattened @ self.weights # log counts
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)
            loss = -probs[torch.arange(self.x_batch.shape[0]), self.y_batch].log().mean() # + (W**2).mean()
            if verbose: print(f"Epoch {i}. Loss: {loss.item():.4f}")

            # Backward pass
            self.weights.grad = None # set the gradient to zero, but more performantly than = 0
            loss.backward()

            # Update the weights
            assert(self.weights.grad is not None)
            self.weights.data += -learn_rate * self.weights.grad
        
        return self.weights, loss.item()

    def generate(self, num:int=10, generator:torch.Generator|None=None, max_length=100, verbose=False):
        """
        Generates `num` sequences of tokens using the trained bigram language model.

        Args:
            num (int): The number of sequences to generate. Defaults to 10.
            generator (torch.Generator, optional): Generator used for random number generation. Defaults to None.
            verbose (bool, optional): If True, prints each generated sequence. Defaults to False.

        Returns:
            list: A list of length `num`, where each element is a list of integers representing the generated sequence.
        """

        output = []
        n = 0
        for i in range(num):
            idxs = [0 for _ in range(3-1)]
            out = ""
            while n < max_length:
                x_enc = F.one_hot(torch.tensor(idxs), self.num_classes).float()
                x_enc_flattened = x_enc.reshape((1,-1))
                logits = x_enc_flattened @ self.weights
                counts = logits.exp()
                probs = counts / counts.sum(1, keepdims=True)

                idx = torch.multinomial(probs, 1, replacement=True, generator=generator)
                # if idx == 0: break
                out += self.decode[idx.item()]
                idxs.append(idx)
                idxs.pop(0)
                n += 1
            output.append(out)
        if verbose: 
            for o in output: print(o)
            
        return output

class NgramLanguageModel():
    """
    A class representing a trigram language model trained with maximum likelihood estimation.

    Attributes:
    - x_data (torch.Tensor): a NxM tensor of integer inputs
    - y_data (torch.Tensor): a 1xM tensor of integer outputs
    - num_classes (int): the number of possible tokens in the input data
    - weights (torch.Tensor): a num_classes x num_classes tensor of neuron weights
    - decode (dict): a dictionary for translating tokens to integers

    Methods:
    - train(epochs, learn_rate, verbose): trains the model on the input data
    - generate(num, generator, verbose): generates num sequences of tokens from the model
    """
    def __init__(self,x_data:torch.Tensor,y_data:torch.Tensor,N=3,num_classes=None,decode=None):
        """
        Initializes a new instance of the BigramLanguageModel class.

        Args:
        - x_data (torch.Tensor): a NxM tensor of integer inputs
        - y_data (torch.Tensor): a 1xM tensor of integer outputs
        - num_classes (int, optional): the number of possible tokens in the input data
        - decode (dict, optional): a dictionary for translating tokens to integers
        """
        self.N = N
        self.x_data = x_data # a 1xM array of integer inputs
        self.y_data = y_data # a 1xM array of integer outputs
        self.x_batch = self.x_data
        self.y_batch = self.y_data

        # Determine the number of possible tokens from the input data
        if num_classes is None:
            num_classes = max([x.item() for x in self.x_data]) + 1
        self.num_classes = num_classes

        self.num_classes = num_classes if num_classes else max([x.item() for x in self.x_data])+1

        # Initialize the neuron weights
        self.weights = torch.zeros((self.x_data.shape[1]*num_classes, num_classes), requires_grad=True)

        # Set the dictionary for token decoding
        self.decode = decode

    def train(self, epochs=100, learn_rate=1e-3, batch_size=None, verbose=False, generator=None):
        """
        Trains the model on the input data.

        Args:
        - epochs (int, optional): the number of training epochs
        - learn_rate (float, optional): the learning rate for the optimizer
        - verbose (bool, optional): whether to print training progress to stdout

        Returns:
        - weights (torch.Tensor): the trained neuron weights
        - loss (float): the final loss value
        """
        for i in range(epochs):

            # Forward pass
            if batch_size is not None:
                # selecting `batch_size` random samples from the x_data
                idxs = torch.multinomial(torch.zeros(len(self.x_data))+1, batch_size, replacement=True, generator=generator)
                self.x_batch = self.x_data[idxs.tolist()] # N x 2 array
                self.y_batch = self.y_data[idxs.tolist()] # N x 1 array
            else:
                self.x_batch = self.x_data # N x 2 array
                self.y_batch = self.y_data # N x 1 array
            
            # forward pass
            x_enc = F.one_hot(self.x_batch, num_classes=self.num_classes).float()
            x_enc_flattened = x_enc.reshape(batch_size,-1)
            # print(self.weights.shape)
            # print(x_enc.shape)
            # print(x_enc_flattened.shape)
            logits = x_enc_flattened @ self.weights # log counts
            counts = logits.exp()
            probs = counts / counts.sum(1, keepdims=True)
            loss = -probs[torch.arange(self.x_batch.shape[0]), self.y_batch].log().mean() # + (W**2).mean()
            if verbose: print(f"Epoch {i}. Loss: {loss.item():.4f}")

            # Backward pass
            self.weights.grad = None # set the gradient to zero, but more performantly than = 0
            loss.backward()

            # Update the weights
            assert(self.weights.grad is not None)
            self.weights.data += -learn_rate * self.weights.grad
        
        return self.weights, loss.item()

    def generate(self, num:int=10, generator:torch.Generator|None=None, max_length=100, verbose=False):
        """
        Generates `num` sequences of tokens using the trained bigram language model.

        Args:
            num (int): The number of sequences to generate. Defaults to 10.
            generator (torch.Generator, optional): Generator used for random number generation. Defaults to None.
            verbose (bool, optional): If True, prints each generated sequence. Defaults to False.

        Returns:
            list: A list of length `num`, where each element is a list of integers representing the generated sequence.
        """

        output = []
        n = 0
        for i in range(num):
            idxs = [0 for _ in range(self.N-1)]
            out = ""
            while n < max_length:
                x_enc = F.one_hot(torch.tensor(idxs), self.num_classes).float()
                x_enc_flattened = x_enc.reshape((1,-1))
                # print("W:", self.weights.shape)
                # print("x_enc:", x_enc.shape)
                # print("x_enc_flat", x_enc_flattened.shape)
                logits = x_enc_flattened @ self.weights
                counts = logits.exp()
                probs = counts / counts.sum(1, keepdims=True)

                idx = torch.multinomial(probs, 1, replacement=True, generator=generator)
                # if idx == 0: break
                out += self.decode[idx.item()]
                idxs.append(idx)
                idxs.pop(0)
                n += 1
            output.append(out)
            if verbose: print(out)
            
        return output