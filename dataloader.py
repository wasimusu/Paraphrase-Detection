import os


def parse_paraphrase_corpus(filename):
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

    return sentencesX, sentencesY, label