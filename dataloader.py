import os
import random


def dataLoader(filename, shuffle=True):
    """
    The corpus contains lots of things which are not of our interest.
    Read file and discard things that are of not interest to us.

    :param filename:
    :return: sentencesX, sentencesY, label
    :return type: list of sentence, list of sentence, list of bools
    :param sentence type : string
    """
    if not os.path.exists(filename):
        raise ValueError("{} file does not exist.")

    # Read file and split into lines
    text = open(filename, mode='r', encoding='utf8').read().lower().splitlines()

    labels, sentenceX, sentenceY = [], [], []
    for line in text[1:]:
        label, _, _, s1, s2 = line.split("\t")

        labels.append(label)
        sentenceX.append(s1)
        sentenceY.append(s2)

    return labels, sentenceX, sentenceY


if __name__ == '__main__':
    filename = "data/msr_paraphrase_test.txt"
    label, sx, sy = dataLoader(filename)
    print(label[0], sx[0], sy[0])
