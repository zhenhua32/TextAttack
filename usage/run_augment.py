"""
先试试回译吧
"""

import os
import sys

sys.path.append("../")

from tqdm import tqdm

from textattack.transformations.sentence_transformations import BackTranslation
from textattack.constraints.pre_transformation import RepeatModification, StopwordModification
from textattack.augmentation import Augmenter


src_dir = r"D:\code\github\py_nlp_classify\data"
train_file = os.path.join(src_dir, "train.csv")
dev_file = os.path.join(src_dir, "dev.csv")


transformation = BackTranslation(
    src_lang="zh",
    target_lang="en",
    src_model="Helsinki-NLP/opus-mt-en-zh",
    target_model="Helsinki-NLP/opus-mt-zh-en",
)
constraints = [RepeatModification(), StopwordModification()]
augmenter = Augmenter(transformation=transformation, constraints=constraints, transformations_per_example=1)


def augment_data(input_file, output_file):
    """
    直接生成一个新的文件
    """
    result = set()
    count = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            count += 1
            result.add(line)
            label, text = line.strip().split("\t")
            augmented_text = augmenter.augment(text)
            new_line = f"{label}\t{augmented_text[0]}\n"
            result.add(new_line)

    print(f"原始数据量: {count}, 增强后数据量: {len(result)}")

    with open(output_file, "w", encoding="utf-8") as f:
        f.writelines(result)


if __name__ == "__main__":
    augment_data(train_file, "train_aug.csv")
    augment_data(dev_file, "dev_aug.csv")
