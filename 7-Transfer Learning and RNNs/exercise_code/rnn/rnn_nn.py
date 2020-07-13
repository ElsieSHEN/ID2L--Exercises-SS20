import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20, activation="tanh"):
        super(RNN, self).__init__()
        
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        - activation: Nonlinearity in cell; 'tanh' or 'relu'
        """
        ############################################################################
        # TODO: Build a simple one layer RNN with an activation with the attributes#
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h as 0 if these values are not given.                          #
        ############################################################################
        
        self.hidden_size = hidden_size
        self.input_size = input_size

        if activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        self.fc1 = nn.Linear(self.input_size, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        

    def forward(self, x, h=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Optional hidden vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []

        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        h = torch.zeros(batch_size, self.hidden_size)

        for i in range(seq_len):
            h = self.activation(self.fc1(x[i]) + self.fc2(h))
            h_seq.append(h)
        
        h_seq = torch.stack(h_seq)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        return h_seq , h

class LSTM(nn.Module):
    def __init__(self, input_size=1 , hidden_size=20):
        super(LSTM, self).__init__()
        """
        Inputs:
        - input_size: Number of features in input vector
        - hidden_size: Dimension of hidden vector
        """
        ############################################################################
        # TODO: Build a one layer LSTM with an activation with the attributes      #
        # defined above and a forward function below. Use the nn.Linear() function #
        # as your linear layers.                                                   #
        # Initialse h and c as 0 if these values are not given.                    #
        ############################################################################
        
        self.hidden_size = hidden_size
        self.input_size = input_size

        self.xf = nn.Linear(input_size, self.hidden_size)
        self.xi = nn.Linear(input_size, self.hidden_size)
        self.xo = nn.Linear(input_size, self.hidden_size)
        self.xg = nn.Linear(input_size, self.hidden_size)

        self.hf = nn.Linear(self.hidden_size, self.hidden_size)
        self.hi = nn.Linear(self.hidden_size, self.hidden_size)
        self.ho = nn.Linear(self.hidden_size, self.hidden_size)
        self.hg = nn.Linear(self.hidden_size, self.hidden_size)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################       


    def forward(self, x, h=None , c=None):
        """
        Inputs:
        - x: Input tensor (seq_len, batch_size, input_size)
        - h: Hidden vector (nr_layers, batch_size, hidden_size)
        - c: Cell state vector (nr_layers, batch_size, hidden_size)

        Outputs:
        - h_seq: Hidden vector along sequence (seq_len, batch_size, hidden_size)
        - h: Final hidden vetor of sequence(1, batch_size, hidden_size)
        - c: Final cell state vetor of sequence(1, batch_size, hidden_size)
        """
        h_seq = []


        #######################################################################
        #  TODO: Perform the forward pass                                     #
        #######################################################################   

        seq_len = x.shape[0]
        batch_size = x.shape[1]

        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)

        for xt in x:
            f = self.sigmoid(self.xf(xt) + self.hf(h))
            i = self.sigmoid(self.xi(xt) + self.hi(h))
            o = self.sigmoid(self.xo(xt) + self.ho(h))
            g = self.tanh(self.xg(xt) + self.hg(h))
            c = torch.mul(f, c) + torch.mul(i, g)
            h = torch.mul(o, self.tanh(c))
            h_seq.append(h)
        
        h_seq = torch.stack(h_seq)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
    
        return h_seq , (h, c)

