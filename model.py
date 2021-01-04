from torch.nn import functional as F
import pytorch_lightning.core as core
import pytorch_lightning.metrics as metrics
import torch.nn as nn
import torch.optim as optim

class RNN(core.LightningModule):

    LR = 1e-4
    IN_SIZE = 500
    HIDDEN_SIZE = 128
    NUM_LAYERS = 3
    DROPOUT = 0.2

    def __init__(self, in_size=IN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.in_size = in_size
        self.accuracy = metrics.Accuracy()

        self.embedding_layer = nn.Embedding(12, 50)
        self.dropout_layer_1 = nn.Dropout(dropout)
        self.linear_layer_1 = nn.Linear(200, 100)
        self.dropout_layer_2 = nn.Dropout(dropout)
        self.linear_layer_2 = nn.Linear(100, 10)
        self.dropout_layer_3 = nn.Dropout(dropout)
        self.gru_layer = nn.GRU(
            in_size, 
            hidden_size, 
            num_layers=num_layers, 
            batch_first=True, 
            bidirectional=True, 
            dropout=dropout
        )
        self.__init_gru()
        self.linear_layer_3 = nn.Linear(2 * hidden_size, 5)

    def forward(self, x):
        x = self.embedding_layer(x)
        
        x = self.dropout_layer_1(x)
        
        x = x.permute((0, 2, 3, 1))
        x = self.linear_layer_1(x)
        x = F.relu(x)
        
        x = self.dropout_layer_2(x)
        
        x = self.linear_layer_2(x)
        x = F.relu(x)
        
        x = self.dropout_layer_3(x)
        
        x = x.reshape(-1, 90, self.in_size)
        x, _ = self.gru_layer(x)
        
        return self.linear_layer_3(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # x, y = x.type(torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor), y.to(device)
        output = self(x).transpose(1, 2)
        train_loss = F.cross_entropy(output, y)
        
        self.log('train_loss', train_loss, prog_bar=True, on_step=batch_idx % 100 == 0, on_epoch=True)
        
        return train_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        # x, y = x.type(torch.cuda.LongTensor if device.type == 'cuda' else torch.LongTensor), y.to(device)
        output = self(x).transpose(1, 2)
        val_loss = F.cross_entropy(output, y)

        self.accuracy(output, y)
        self.log('val_loss', val_loss, on_step=False, on_epoch=True)
        self.log('val_acc', self.accuracy, on_step=False, on_epoch=True)
        
        return val_loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=RNN.LR)

    def __init_gru(self):
        for parameter in self.gru_layer.parameters():
            if len(parameter.shape) >= 2:
                nn.init.orthogonal_(parameter.data)
            else:
                nn.init.normal_(parameter.data)

