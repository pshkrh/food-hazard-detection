import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertModel


class EnhancedClassifier(nn.Module):
    """
    A custom classifier that leverages a pre-trained BERT model for text classification tasks.

    This classifier adds a dropout layer and a linear layer on top of the BERT base model to perform
    classification into a specified number of labels. It's designed to fine-tune BERT for improved performance
    on classification tasks.
    """

    def __init__(self, model_name, num_labels, dropout_rate=0.3):
        """
        Initializes the EnhancedClassifier with a pre-trained BERT model, dropout layer, and classification layer.

        Parameters:
            model_name (str): The name of the pre-trained BERT model to use.
            num_labels (int): The number of output labels/classes for classification.
            dropout_rate (float, optional): The dropout rate for regularization. Defaults to 0.3.
        """
        super().__init__()
        self.base_model = BertModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        """
        Performs a forward pass through the classifier.

        Parameters:
            input_ids (torch.Tensor): Tensor of input token IDs.
            attention_mask (torch.Tensor): Tensor indicating which tokens should be attended to.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        outputs = self.base_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.classifier(x)
        return logits


class DNNClassifier(nn.Module):
    """
    A Deep Neural Network (DNN) classifier for text classification tasks.

    This model consists of an embedding layer, a fully connected hidden layer with ReLU activation,
    a dropout layer for regularization, and an output layer for classification.
    """

    def __init__(self, vocab_sz, num_lbls, emb_dim=100, hid_dim=256, drop_val=0.3):
        """
        Initializes the DNNClassifier with embedding, hidden, dropout, and output layers.

        Parameters:
            vocab_sz (int): The size of the vocabulary.
            num_lbls (int): The number of output labels/classes for classification.
            emb_dim (int, optional): The dimensionality of the embedding layer. Defaults to 100.
            hid_dim (int, optional): The number of neurons in the hidden layer. Defaults to 256.
            drop_val (float, optional): The dropout rate for regularization. Defaults to 0.3.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hid_dim)
        self.dropout = nn.Dropout(drop_val)
        self.fc2 = nn.Linear(hid_dim, num_lbls)

    def forward(self, input_ids):
        """
        Performs a forward pass through the DNN classifier.

        Parameters:
            input_ids (torch.Tensor): Tensor of input token IDs.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        emb = self.embedding(input_ids)
        emb_avg = torch.mean(emb, dim=1)
        hidden = F.relu(self.fc1(emb_avg))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits


class DANClassifier(nn.Module):
    """
    A Deep Averaging Network (DAN) classifier for text classification tasks.

    This model averages word embeddings and passes them through fully connected layers with ReLU activation
    and dropout for classification.
    """

    def __init__(self, vocab_sz, num_lbls, emb_dim=100, hid_dim=256, drop_val=0.3):
        """
        Initializes the DANClassifier with embedding, hidden, dropout, and output layers.

        Parameters:
            vocab_sz (int): The size of the vocabulary.
            num_lbls (int): The number of output labels/classes for classification.
            emb_dim (int, optional): The dimensionality of the embedding layer. Defaults to 100.
            hid_dim (int, optional): The number of neurons in the hidden layer. Defaults to 256.
            drop_val (float, optional): The dropout rate for regularization. Defaults to 0.3.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim)
        self.fc1 = nn.Linear(emb_dim, hid_dim)
        self.dropout = nn.Dropout(drop_val)
        self.fc2 = nn.Linear(hid_dim, num_lbls)

    def forward(self, input_ids):
        """
        Performs a forward pass through the DAN classifier.

        Parameters:
            input_ids (torch.Tensor): Tensor of input token IDs.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        emb = self.embedding(input_ids)
        emb_avg = torch.mean(emb, dim=1)
        hidden = F.relu(self.fc1(emb_avg))
        hidden = self.dropout(hidden)
        logits = self.fc2(hidden)
        return logits


class CNNClassifier(nn.Module):
    """
    A Convolutional Neural Network (CNN) classifier for text classification tasks.

    This model uses multiple convolutional filters with varying kernel sizes to capture different n-gram features,
    followed by max pooling, dropout, and a fully connected output layer for classification.
    """

    def __init__(
        self,
        vocab_sz,
        num_lbls,
        emb_dim=100,
        nfilt=100,
        f_sizes=[3, 4, 5],
        drop_val=0.3,
    ):
        """
        Initializes the CNNClassifier with embedding, convolutional, dropout, and output layers.

        Parameters:
            vocab_sz (int): The size of the vocabulary.
            num_lbls (int): The number of output labels/classes for classification.
            emb_dim (int, optional): The dimensionality of the embedding layer. Defaults to 100.
            nfilt (int, optional): The number of filters per convolutional layer. Defaults to 100.
            f_sizes (list, optional): A list of kernel sizes for convolutional layers. Defaults to [3, 4, 5].
            drop_val (float, optional): The dropout rate for regularization. Defaults to 0.3.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_sz, emb_dim)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, nfilt, (fs, emb_dim)) for fs in f_sizes]
        )
        self.dropout = nn.Dropout(drop_val)
        self.fc = nn.Linear(nfilt * len(f_sizes), num_lbls)

    def forward(self, input_ids):
        """
        Performs a forward pass through the CNN classifier.

        Parameters:
            input_ids (torch.Tensor): Tensor of input token IDs.

        Returns:
            torch.Tensor: The output logits for each class.
        """
        emb = self.embedding(input_ids)
        emb = emb.unsqueeze(1)
        conv_outs = []
        for conv in self.convs:
            conv_out = F.relu(conv(emb))
            conv_out = conv_out.squeeze(3)
            conv_out = F.max_pool1d(conv_out, conv_out.size(2))
            conv_out = conv_out.squeeze(2)
            conv_outs.append(conv_out)
        concat_out = torch.cat(conv_outs, dim=1)
        concat_out = self.dropout(concat_out)
        logits = self.fc(concat_out)
        return logits
