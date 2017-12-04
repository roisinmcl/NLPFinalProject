from collections import Counter

class Eval:
    def __init__(self, gold, pred):
        assert len(gold)==len(pred)
        self.gold = gold
        self.pred = pred

    def accuracy(self):
        numer = sum(1 for p,g in zip(self.pred,self.gold) if p==g)
        return numer / len(self.gold)

    def precision(self, label):
    # True positives / (true positives + false positivies)
        true_positives = 0
        for i in range(0, len(self.gold)):
            if (self.gold[i] == label) and (self.pred[i] == label):
                true_positives += 1
        false_positives = 0
        for i in range(0, len(self.gold)):
            if (self.gold[i] != label) and (self.pred[i] == label):
                false_positives += 1

        if (true_positives + false_positives) > 0:
            return true_positives / (true_positives + false_positives)
        else:
            return 0

    def recall(self, label):
        # True positives / (true positives + false negatives)
        true_positives = 0
        for i in range(0, len(self.gold)):
            if (self.gold[i] == label) and (self.pred[i] == label):
                true_positives += 1
        false_negatives = 0
        for i in range(0, len(self.gold)):
            if (self.gold[i] == label) and (self.pred[i] != label):
                false_negatives += 1

        if (true_positives + false_negatives) > 0:
            return true_positives / (true_positives + false_negatives)
        else:
            return 0

    def f1(self, label):
        precision = self.precision(label)
        recall = self.recall(label)
        if (precision + recall) > 0:
            return 2 * precision * recall / (precision + recall)
        else:
            return 0

    def confusion_matrix(self):
    	label_indicies = {}
    	labels = ['R', 'D', 'I']
    	ind = 0
    	for label in labels:
    		label_indicies[label] = ind
    		ind += 1

    	matrix = [[0 for x in range(len(labels))] for y in range(len(labels))] 
    	for i in range(len(self.gold)):
    		row = label_indicies[self.pred[i]]
    		col = label_indicies[self.gold[i]]
    		matrix[row][col] += 1

    	return matrix




