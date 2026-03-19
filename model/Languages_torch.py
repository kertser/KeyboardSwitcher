# This library us a language toolset for the project.

# import the necessary packages
import torch
import torch.nn as nn
import pickle

russian_layout = 'ёйцукенгшщзхъфывапролджэ\ячсмитьбю.ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭ/ЯЧСМИТЬБЮ,'
english_layout = '''`qwertyuiop[]asdfghjkl;'\zxcvbnm,./~QWERTYUIOP{}ASDFGHJKL:"|ZXCVBNM<>?'''
hebrew_layout = ";/'קראטוןםפ][שדגכעיחלךף,\זסבהנמצתץ."
special_characters = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
numbers = "0123456789"

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
def predict_language(text, model, char_to_index, max_length, device):
    # Tokenization of text and converting characters to indices
    input_indices = [char_to_index.get(char, 0) for char in text if char in char_to_index]

    # Adding padding to the input
    if len(input_indices) < max_length:
        num_padding = max_length - len(input_indices)
        input_indices += [0] * num_padding

    # Convert into the pytorch tensor and change the dimension
    input_tensor = torch.tensor(input_indices).view(1, -1).to(device)

    # Deliver the input to the model
    with torch.no_grad():
        model.eval()  # Set the model to evaluation mode
        output = model(input_tensor)

    # Generating the prediction
    _, predicted_class = torch.max(output, 1)

    # Вернуть класс языка на основе предсказания
    if predicted_class.item() == 0:
        return None
    elif predicted_class.item() == 1:
        return "en"
    elif predicted_class.item() == 2:
        return "he"
    elif predicted_class.item() == 3:
        return "ru"

# Helper function to create a conversion map
def create_conversion_map(source_layout, target_layout):
    conversion_map = {}

    for src_char, tgt_char in zip(source_layout, target_layout):
        conversion_map[ord(src_char)] = ord(tgt_char)
    return conversion_map

#  Helper function to convert text
def convert_text(text, conversion_map):
    return text.translate(conversion_map)


# Main function to convert text from one layout to another
def convert_text_bidirectional(text, from_layout, to_layout):
    if to_layout == hebrew_layout:
        text = text.lower()
    return convert_text(text, create_conversion_map(from_layout, to_layout))

#print(convert_text_bidirectional("Hello", english_layout, hebrew_layout))

"""
## ???
russian_to_english_map = create_conversion_map(russian_layout, english_layout)
russian_to_hebrew_map = create_conversion_map(russian_layout, hebrew_layout)
english_to_russian_map = create_conversion_map(english_layout, russian_layout)
english_to_hebrew_map = create_conversion_map(english_layout, hebrew_layout)
hebrew_to_russian_map = create_conversion_map(hebrew_layout, russian_layout)
hebrew_to_english_map = create_conversion_map(hebrew_layout, english_layout)
"""

if __name__ == '__main__':
    # Example"
    input_text = "привет"
    args = load_model()
    predicted_language = predict_language(input_text, *args)

    print("Предсказанный язык:", predicted_language)