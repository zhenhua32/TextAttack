"""
BackTranslation class
-----------------------------------

"""


import random

import torch
from transformers import MarianMTModel, MarianTokenizer

from textattack.shared import AttackedText

from .sentence_transformation import SentenceTransformation


class BackTranslation(SentenceTransformation):
    """回译, 需要支持 GPU 加速, 不然太慢了
    A type of sentence level transformation that takes in a text input,
    translates it into target language and translates it back to source
    language.

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)

    src_lang (string): source language
    target_lang (string): target language, for the list of supported language check bottom of this page
    src_model: translation model from huggingface that translates from source language to target language
    target_model: translation model from huggingface that translates from target language to source language
    chained_back_translation: run back translation in a chain for more perturbation (for example, en-es-en-fr-en)

    Example::

        >>> from textattack.transformations.sentence_transformations import BackTranslation
        >>> from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
        >>> from textattack.augmentation import Augmenter

        >>> transformation = BackTranslation()
        >>> constraints = [RepeatModification(), StopwordModification()]
        >>> augmenter = Augmenter(transformation = transformation, constraints = constraints)
        >>> s = 'What on earth are you doing here.'

        >>> augmenter.augment(s)
    """

    def __init__(
        self,
        src_lang="en",
        target_lang="es",
        src_model="Helsinki-NLP/opus-mt-ROMANCE-en",
        target_model="Helsinki-NLP/opus-mt-en-ROMANCE",
        chained_back_translation=0,
        device=None,
    ):
        self.src_lang = src_lang
        self.target_lang = target_lang
        self.target_model = MarianMTModel.from_pretrained(target_model)
        self.target_tokenizer = MarianTokenizer.from_pretrained(target_model)
        self.src_model = MarianMTModel.from_pretrained(src_model)
        self.src_tokenizer = MarianTokenizer.from_pretrained(src_model)
        self.chained_back_translation = chained_back_translation

        # GPU 加速
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.target_model.to(self.device)
        self.src_model.to(self.device)

    def translate(self, input: list, model: MarianMTModel, tokenizer: MarianTokenizer, lang="es"):
        """
        执行单次翻译
        input: 是个 list, 但是只会用到第一个元素
        """
        # change the text to model's format
        src_texts = []
        if lang == "en":
            src_texts.append(input[0])
        else:
            if ">>" and "<<" not in lang:
                lang = ">>" + lang + "<< "
            src_texts.append(lang + input[0])

        # tokenize the input
        encoded_input = tokenizer(src_texts, return_tensors="pt").to(self.device)

        # translate the input
        # 直接根据长度, 来个动态的好了
        max_new_tokens = min(512, encoded_input["input_ids"].shape[1] * 2)
        translated = model.generate(**encoded_input, max_new_tokens=max_new_tokens)
        translated_input = tokenizer.batch_decode(translated, skip_special_tokens=True)
        return translated_input

    def _get_transformations(self, current_text, indices_to_modify):
        transformed_texts = []
        current_text = current_text.text

        # 这个有条件限制的, 需要 supported_language_codes 大于 chained_back_translation
        # 如果 chained_back_translation 不为 0, 则进行多次回译
        # to perform chained back translation, a random list of target languages are selected from the provided model
        if self.chained_back_translation:
            list_of_target_lang = random.sample(
                self.target_tokenizer.supported_language_codes,
                self.chained_back_translation,
            )
            for target_lang in list_of_target_lang:
                # 先使用 target_model 翻译成目标语言, 再使用 src_model 翻译回源语言
                target_language_text = self.translate(
                    [current_text],
                    self.target_model,
                    self.target_tokenizer,
                    target_lang,
                )
                src_language_text = self.translate(
                    target_language_text,
                    self.src_model,
                    self.src_tokenizer,
                    self.src_lang,
                )
                current_text = src_language_text[0]
            return [AttackedText(current_text)]

        # 只进行一次回译
        # translates source to target language and back to source language (single back translation)
        target_language_text = self.translate(
            [current_text], self.target_model, self.target_tokenizer, self.target_lang
        )
        src_language_text = self.translate(
            target_language_text, self.src_model, self.src_tokenizer, self.src_lang
        )
        transformed_texts.append(AttackedText(src_language_text[0]))
        return transformed_texts


"""
List of supported languages
['fr',
 'es',
 'it',
 'pt',
 'pt_br',
 'ro',
 'ca',
 'gl',
 'pt_BR<<',
 'la<<',
 'wa<<',
 'fur<<',
 'oc<<',
 'fr_CA<<',
 'sc<<',
 'es_ES',
 'es_MX',
 'es_AR',
 'es_PR',
 'es_UY',
 'es_CL',
 'es_CO',
 'es_CR',
 'es_GT',
 'es_HN',
 'es_NI',
 'es_PA',
 'es_PE',
 'es_VE',
 'es_DO',
 'es_EC',
 'es_SV',
 'an',
 'pt_PT',
 'frp',
 'lad',
 'vec',
 'fr_FR',
 'co',
 'it_IT',
 'lld',
 'lij',
 'lmo',
 'nap',
 'rm',
 'scn',
 'mwl']
"""
