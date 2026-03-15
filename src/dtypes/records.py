class Example:
    def __init__(self, art, summ, art_ids=None, sum_ids=None):
        self.art = art
        self.summ = summ
        self.art_ids = art_ids
        self.sum_ids = sum_ids


class Batch:
    def __init__(self, art_ids, sum_ids):
        self.art_ids = art_ids
        self.sum_ids = sum_ids
