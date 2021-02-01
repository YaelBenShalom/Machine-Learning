import numpy
import math

class Node():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None, target = None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value
        self.branches = [] if branches is None else branches
        # target attribute will represent a leaf that is the final prediction result
        self.target = target

class DecisionTree():
    def __init__(self, attribute_names):
        """
        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)
        
        """
        self.attribute_names = attribute_names
        self.tree = Node()
        

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def fit_recursion(self, node, features, targets):     
        max_info_index = find_max_information_gain(features, targets)

        values_for_index = set((features[:,max_info_index])) 
        if len(values_for_index) == 1:
            # No added value for maximal information gain -> we reached our stop point
            target_label = get_most_common_label(targets)
            n = Node(target_label, "Result", None, [], target_label)
            node.branches = [n]
            return 

        positive_attributes, negative_attributes, positive_attributes_targets, negative_attributes_targets = split_tables(features, max_info_index, targets)
        
        node_positive = Node(1, self.attribute_names[max_info_index], max_info_index, [], None)
        node_negative = Node(0, self.attribute_names[max_info_index], max_info_index, [], None)
        node.branches = [node_negative, node_positive]

        # Now in recursion for both 0 and 1 results 
        self.fit_recursion(node_negative, negative_attributes, negative_attributes_targets)
        self.fit_recursion(node_positive, positive_attributes, positive_attributes_targets)

    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        Output:
            VOID: It should update self.tree with a built decision tree.
        """
        self._check_input(features)
        self.fit_recursion(self.tree, features, targets)

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        Outputs:
            predictions (np.array): numpy array of size N array which has the predicitons 
            for the input data.
        """
    
        self._check_input(features)
        predictions = numpy.zeros((len(features), 1))
        for i in range(len(features)):
            current_node = self.tree
            while len(current_node.branches) != 1:
                # If we've reached the 0/1 target value node
                next_node = current_node.branches[0]
                true_value = features[i][next_node.attribute_index]
                if true_value == 0:
                    # Go negative
                    current_node = current_node.branches[0]
                else:
                    # Go positive
                    current_node = current_node.branches[1]
            predicted_value = current_node.branches[0].target
            predictions[i] = predicted_value
        
        return predictions


    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)


def entropy(column):
    """ Caluclate the entropy of a column """
    total = len(column)
    p = column.sum()
    n = total - p
    if n == 0 or p == 0:
        return 0
    tot_entropy = -(p/total)*math.log(p/total, 2) - (n/total)*math.log(n/total, 2)
    return tot_entropy


def split_tables(features, attribute_index, targets):
    """ Split the tables of features and targets by the attribute index values """
    attribute_col = features[:, attribute_index]
    ones_counter = int(attribute_col.sum())
    positive_attributes = numpy.zeros((ones_counter, len(features[0])))
    negative_attributes = numpy.zeros((features.shape[0] - ones_counter, len(features[0])))
    positive_attributes_targets = numpy.zeros((ones_counter, 1))
    negative_attributes_targets = numpy.zeros((features.shape[0] - ones_counter, 1))

    counter_positive = 0
    counter_negative = 0
    for i in range(len(features)):
        if features[i, attribute_index] == 0:
            # Negative attribute
            negative_attributes[counter_negative] = features[i]
            negative_attributes_targets[counter_negative] = targets[i]
            counter_negative += 1
        else:
            # Positive attribute
            positive_attributes[counter_positive] = features[i]
            positive_attributes_targets[counter_positive] = targets[i]
            counter_positive += 1
    return positive_attributes, negative_attributes, positive_attributes_targets, negative_attributes_targets


def information_gain(features, attribute_index, targets):
    """
    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """

    positive_attributes, negative_attributes, positive_attributes_targets, negative_attributes_targets = split_tables(features, attribute_index, targets)
    pp = positive_attributes_targets.sum()
    pn = positive_attributes_targets.shape[0] - pp
    pe = entropy(positive_attributes_targets)

    np = negative_attributes_targets.sum()
    nn = negative_attributes_targets.shape[0] - np
    ne = entropy(negative_attributes_targets)

    tot = pp + pn + np + nn
    average_entropy = ((pp + pn)/tot)*pe + ((np + nn)/tot)*ne

    total_entropy = entropy(targets)
    return total_entropy - average_entropy

def find_max_information_gain(features, targets):
    """ return index of maximal info gain for features"""
    max_info_index = 0
    max_information_gain = 0
    for i in range(len(features[0])):    
        if information_gain(features, i, targets) > max_information_gain:
            max_info_index = i
            max_information_gain = information_gain(features, i, targets)
    return max_info_index


def get_most_common_label(targets):
    """ get the most frequent target - 0 or 1 """ 
    ones_counter = int(targets.sum())
    if len(targets)/2 > ones_counter:
        return 0
    return 1
    

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
