import re
from nltk import word_tokenize
from nltk.corpus import stopwords


class PreProcessing:
    stop = set(stopwords.words('english'))

    @staticmethod
    def convert_to_lower_case(text):
        try:
            return str(text).lower()
        except TypeError:
            print("Invalid String Provided")
            return text

    @staticmethod
    def removes_special_char(text):
        try:
            pattern = r'[?|$|.|!|,|@|(|)|:|\n|\t|"|¦|\]|\[|{|}|*|~|;|♥|•|=|^|#|%|]'
            filtered_text = re.sub(pattern, ' ', text)
            return filtered_text.strip()
        except Exception as e:
            print(e)
            return text

    @staticmethod
    def replace_multiple_spaces_with_single(text):
        pattern = r"\s\s+"
        filtered_text = re.sub(pattern, " ", text)
        return filtered_text

    @staticmethod
    def remove_stop_words(text):
        return ' '.join([i for i in word_tokenize(text) if i not in PreProcessing.stop])
