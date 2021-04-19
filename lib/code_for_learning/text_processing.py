import re

from pymystem3 import Mystem


class TextNormalizer(object):
    DOMENS = {".ru", ".com", ".ру", ".рф", ".su", ".ju", ".r", ".net", ".io", ".org", ".gov", ".info", ".juc"}

    def __init__(self, names, digits):
        """
        Class for text normalization (stemming, remove noise words,  refactor: phone numbers, emails etc.)
        :param names: Dict with russian given first names
        :param digits: Dict with alphabetic numbers mapping
        """
        self.stem = Mystem()
        self.re_number = re.compile("(\d+)")

        # https://betterprogramming.pub/detecting-external-links-in-a-paragraph-of-text-with-javascript-automatically-3c15537f4997
        self.re_url = re.compile("(https?:\/\/)?([@\w\-~]+)(\.[\w\-~]+)+(\/[\w\-~@:%]*)*(#[\w\-]*)?(\?[^\s]*)?")
        self.names = names
        self.digits = digits

    def replace_url(self, text):
        for url in self.re_url.finditer(text):
            if url.group(3) in self.DOMENS:
                if url.group(2).find("@") >= 0:
                    text = text.replace(url.group(), "TOKENEMAIL")
                else:
                    text = text.replace(url.group(), "TOKENWEB")
        return text

    def tokenize(self, text):
        for token in self.stem.lemmatize(text):
            token = re.sub("[^0-9а-яА-Яa-zA-Z@#]+", "", token)
            if token in self.digits:
                yield str(self.digits[token])
            elif token in self.names:
                yield "TOKENNAME"
            #             token = re.sub("[^0-9а-яА-Яa-zA-Z@#]+", "", token)
            elif (not token.isalpha()) or (len(token) > 2) or (token in self.DOMENS):
                yield token
            elif token in self.DOMENS:
                yield "TOKENWEB"

    def normal_generator(self, text_col):
        for text in text_col:
            text = text.lower()
            text = self.replace_url(text)
            text = self.re_number.sub(" \g<1> ", text)
            text = " ".join(self.tokenize(text))
            text = re.sub(r"(\d+)\s+(?=\d+\b)", lambda x: x.group(1) if len(x.group()) <= 11 else " " + x.group(), text)
            text = self.re_number.sub(lambda x: "TOKENNUM{}".format(len(x.group())), text)
            yield text

    def fit_transform(self, X):
        X["text"] = list(self.normal_generator(X["description"] + " " + X["title"]))
        return X
