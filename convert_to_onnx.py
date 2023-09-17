# Convert the torch model to ONNX format

# import the necessary packages
import torch
import torch.nn as nn
import torch.onnx as onnx
import pickle

class LanguageClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LanguageClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Изменяем размерность входных данных
        x = x.view(x.size(0), -1)  # Приводим к размерности (batch_size, sequence_length)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(self.embedding(x), (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Loading the model
def load_model():
    # Load the dictionary of symbols
    with open('dictionary.pkl', 'rb') as file:
        char_to_index = pickle.load(file)

    # Hyperparameters
    max_length = 45  # Constant to the selected model
    input_size = len(char_to_index)  # Размер словаря (количество уникальных символов)
    hidden_size = 64  # Размер скрытого состояния LSTM
    num_layers = 3  # Количество слоев LSTM
    num_classes = 4  # Количество классов (количество языков) включая 0 = нет языка

    # Set-up the device and load the model
    device = torch.device('cpu')
    model = LanguageClassifier(input_size, hidden_size, num_layers, num_classes)
    model.load_state_dict(torch.load('lang_model.pth'))
    model.to(device)
    args = [model, char_to_index, max_length, device]
    return args

# AI assisted language detection - hebrew,english,russian
def convert_model(model, char_to_index, max_length, device):
    model.eval()  # Set the model to evaluation mode
    onnx_path = "lang_model.onnx"

    # Example input text
    text = "dummy_text"

    # Tokenization of text and converting characters to indices
    input_indices = [char_to_index.get(char, 0) for char in text if char in char_to_index]

    # Adding left-padding to the input if needed
    if len(input_indices) < max_length:
        num_padding = max_length - len(input_indices)
        input_indices = [0] * num_padding + input_indices

    # Convert into a PyTorch tensor and change the dimension
    input_tensor = torch.tensor(input_indices).view(1, -1).to(device)

    # Export the model to ONNX with initial states
    torch.onnx.export(model, input_tensor, onnx_path, verbose=True)

# Convert the model to ONNX format
args = load_model()
convert_model(*args)
