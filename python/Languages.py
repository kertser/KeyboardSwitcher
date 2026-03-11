# This library us a language toolset for the project.

# import the necessary packages
import onnxruntime as ort
import pickle
import numpy as np
import os

russian_layout = 'ёйцукенгшщзхъфывапролджэ\ячсмитьбю.ЁЙЦУКЕНГШЩЗХЪФЫВАПРОЛДЖЭ/ЯЧСМИТЬБЮ,'
english_layout = '''`qwertyuiop[]asdfghjkl;'\zxcvbnm,./~QWERTYUIOP{}ASDFGHJKL:"|ZXCVBNM<>?'''
hebrew_layout = ";/'קראטוןםפ][שדגכעיחלךף,\זסבהנמצתץ."
special_characters = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ "
numbers = "0123456789"


# Loading the model, converted to onnx format
def load_model():
    # Load the ONNX model
    current_directory = os.path.dirname(os.path.realpath(__file__))
    onnx_model_path = os.path.join(current_directory, "lang_model.onnx")
    dictionary_path = os.path.join(current_directory, "dictionary.pkl")
    ort_session = ort.InferenceSession(onnx_model_path)

    with open(dictionary_path, 'rb') as file:
        char_to_index = pickle.load(file)

    return [ort_session, char_to_index, 45]


# AI assisted language detection - hebrew,english,russian
def predict_language(text, ort_session, char_to_index, max_length):
    # Tokenization of text and converting characters to indices

    input_indices = [char_to_index.get(char, 0) for char in text if char in char_to_index]

    # Adding padding to the input
    if len(input_indices) < max_length:
        num_padding = max_length - len(input_indices)
        input_indices += [0] * num_padding

    # Convert into the pytorch tensor and change the dimension
    input_tensor = np.array(input_indices, dtype=np.int64).reshape(1, -1)

    # Run inference
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)

    # The result will be in ort_outputs
    output_probs = ort_outputs[0]

    #Get the predicted class (assuming it's a classification model)
    predicted_class = np.argmax(output_probs)

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

if __name__ == '__main__':
    # Example"
    input_text = "hello"
    args = load_model()

    predicted_language = predict_language(input_text, *args)
    print("Предсказанный язык:", predicted_language)