"""
Word Swap
-------------------------------
Word swap transformations act by replacing some words in the input. Subclasses can implement the abstract ``WordSwap`` class by overriding ``self._get_replacement_words``

"""
import random
import string

from textattack.transformations import Transformation


class WordSwap(Transformation):
    """An abstract class that takes a sentence and transforms it by replacing
    some of its words.
    单词替换抽象类，接受一个句子并通过替换其中的一些单词来转换它。

    letters_to_insert (string): letters allowed for insertion into words
    (used by some char-based transformations)
    """

    def __init__(self, letters_to_insert=None):
        self.letters_to_insert = letters_to_insert
        if not self.letters_to_insert:
            # 默认是小写字母和大写字母, 这会不会是不出中文的原因?
            self.letters_to_insert = string.ascii_letters

    def _get_replacement_words(self, word):
        """Returns a set of replacements given an input word. Must be overriden
        by specific word swap transformations.

        Args:
            word: The input word to find replacements for.
        """
        raise NotImplementedError()

    def _get_random_letter(self):
        """Helper function that returns a random single letter from the English
        alphabet that could be lowercase or uppercase."""
        return random.choice(self.letters_to_insert)

    def _get_transformations(self, current_text, indices_to_modify):
        words = current_text.words
        transformed_texts = []

        for i in indices_to_modify:
            word_to_replace = words[i]
            replacement_words = self._get_replacement_words(word_to_replace)
            transformed_texts_idx = []
            for r in replacement_words:
                if r == word_to_replace:
                    continue
                # 替换单词
                transformed_texts_idx.append(current_text.replace_word_at_index(i, r))
            transformed_texts.extend(transformed_texts_idx)

        return transformed_texts
