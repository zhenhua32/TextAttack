"""
Transformation Abstract Class
============================================

"""

from abc import ABC, abstractmethod

from textattack.shared.utils import ReprMixin
from textattack.shared import AttackedText


class Transformation(ReprMixin, ABC):
    """An abstract class for transforming a sequence of text to produce a
    potential adversarial example.
    生成潜在的对抗样本的文本序列的抽象类
    """

    def __call__(
        self,
        current_text: AttackedText,
        pre_transformation_constraints=[],
        indices_to_modify=None,
        shifted_idxs=False,
        return_indices=False,
    ):
        """
        这个方法本质上是对单个文本进行处理的, 我需要一个 batch 版的, 因为有些数据增强用到了模型, 单次跑效率太低了
        Returns a list of all possible transformations for ``current_text``.
        Applies the ``pre_transformation_constraints`` then calls
        ``_get_transformations``.

        Args:
            current_text: The ``AttackedText`` to transform.
            pre_transformation_constraints: The ``PreTransformationConstraint`` to apply before
                beginning the transformation.
            indices_to_modify: Which word indices should be modified as dictated by the
                ``SearchMethod``.
            shifted_idxs (bool): Whether indices could have been shifted from
                their original position in the text.
            return_indices (bool): Whether the function returns indices_to_modify
                instead of the transformed_texts.
        """
        if indices_to_modify is None:
            indices_to_modify = set(range(len(current_text.words)))
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            indices_to_modify = set(indices_to_modify)

        if shifted_idxs:
            indices_to_modify = set(current_text.convert_from_original_idxs(indices_to_modify))

        # 使用所有的约束条件
        for constraint in pre_transformation_constraints:
            # 取交集
            indices_to_modify = indices_to_modify & constraint(current_text, self)

        if return_indices:
            return indices_to_modify

        # 生成变换后的文本, 核心函数是 _get_transformations
        transformed_texts = self._get_transformations(current_text, indices_to_modify)
        for text in transformed_texts:
            text.attack_attrs["last_transformation"] = self
        return transformed_texts

    @abstractmethod
    def _get_transformations(self, current_text, indices_to_modify: set):
        """Returns a list of all possible transformations for ``current_text``,
        only modifying ``indices_to_modify``. Must be overridden by specific
        transformations.

        Args:
            current_text: The ``AttackedText`` to transform.
            indicies_to_modify: Which word indices can be modified.
        """
        raise NotImplementedError()

    def batch_call(
        self,
        batch_current_text: list,
        pre_transformation_constraints=[],
        batch_indices_to_modify: list = None,
        shifted_idxs=False,
        return_indices=False,
    ):
        if batch_indices_to_modify is None:
            batch_indices_to_modify = [set(range(len(text.words))) for text in batch_current_text]
            # If we are modifying all indices, we don't care if some of the indices might have been shifted.
            shifted_idxs = False
        else:
            batch_indices_to_modify = [set(indices_to_modify) for indices_to_modify in batch_indices_to_modify]

        if shifted_idxs:
            temp = []
            for current_text, indices_to_modify in zip(batch_current_text, batch_indices_to_modify):
                temp.append(set(current_text.convert_from_original_idxs(indices_to_modify)))
            batch_indices_to_modify = temp

        # 使用所有的约束条件
        for constraint in pre_transformation_constraints:
            # 取交集
            batch_indices_to_modify = [
                indices_to_modify & constraint(current_text, self)
                for current_text, indices_to_modify in zip(batch_current_text, batch_indices_to_modify)
            ]

        if return_indices:
            return batch_indices_to_modify

        # 生成变换后的文本, batch_transformed_texts 是个 list 的 list
        batch_transformed_texts = self._get_batch_transformations(batch_current_text, batch_indices_to_modify)
        for transformed_texts in batch_transformed_texts:
            for text in transformed_texts:
                text.attack_attrs["last_transformation"] = self
        return batch_transformed_texts

    @abstractmethod
    def _get_batch_transformations(self, batch_current_text: list, batch_indices_to_modify: list):
        """
        batch_current_text: list[AttackedText]
        batch_indices_to_modify: list[set]

        return: list[list[AttackedText]]
        """
        raise NotImplementedError()

    @property
    def deterministic(self):
        return True
