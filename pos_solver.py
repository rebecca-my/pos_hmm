###################################
#
# code by Rebecca Myers rsciagli
#
# (Based on skeleton code by D. Crandall)
#

import numpy as np
from collections import Counter, defaultdict
import random
import math


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, sentence, label):
        log_prob = 0
        if model == "Simple":
            for word, pos in zip(sentence,label):
                log_prob += np.log(self.emission_dict[pos][word] / self.emission_counts[pos])
            return log_prob
        elif model == "HMM":
            state = 'start'
            for word, pos in zip(sentence,label):
                emission_prob = self.emission_dict[pos][word] / self.emission_counts[pos]
                transition_prob = self.transition_dict[state][pos] / self.transition_counts[state]
                log_prob += np.log(emission_prob) + np.log(transition_prob)
                state = pos
            transition_prob = self.transition_dict[state]['end'] / self.transition_counts[state]
            log_prob += np.log(transition_prob)
            return log_prob
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        #pass
        self.emission_dict = defaultdict(list)
        self.transition_dict = defaultdict(list)
        for line in data:
            #print(line)
            #loops through all word, pos pairs.  output is two dictionaries: emissions, and transitions
            state = 'start'
            for word, pos in zip(line[0],line[1]):
                #print(word,pos)
                self.emission_dict[pos].append(word)
                self.transition_dict[state].append(pos)
                state = pos

            self.transition_dict[state].append('end')

        self.emission_counts = {}
        for key, value in self.emission_dict.items():
            self.emission_counts[key] = len(value)
            self.emission_dict[key] = Counter(value)
            
        self.transition_counts = {}
        for key, value in self.transition_dict.items():
            self.transition_counts[key] = len(value)
            self.transition_dict[key] = Counter(value)

        self.transition_dict = dict(self.transition_dict)
        self.emission_dict = dict(self.emission_dict)

    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence): 
        self.possible_pos = self.emission_counts.keys()
        results = []

        for word in sentence:
            best_pos = 'noun' ##default if word has not been seen before.
            best_prob = 0
            for pos in self.possible_pos:
                prob = self.emission_dict[pos][word] / self.emission_counts[pos]
                if prob > best_prob:
                    best_prob = prob
                    best_pos = pos
            results.append(best_pos)
        return results

    def hmm_viterbi(self, sentence):
        viterbi_prob = []
        viterbi_path = []
        viterbi_prob.append({'start' : 0})
        for word in sentence: 
            prob_dict = {}
            path_dict = {}
            for pos in self.possible_pos:
                best_state = None
                best_prob = -np.Inf
                for state in viterbi_prob[-1].keys():
                    emission_prob = self.emission_dict[pos][word] / self.emission_counts[pos]
                    emission_prob = max(emission_prob, 0.000001)
                    transition_prob = self.transition_dict[state][pos] / self.transition_counts[state]
                    log_prob = np.log(emission_prob) + np.log(transition_prob)
                    log_prob += viterbi_prob[-1][state]
                    if log_prob > best_prob:
                        best_prob = log_prob
                        best_state = state
                prob_dict[pos] = best_prob
                path_dict[pos] = best_state
            viterbi_prob.append(prob_dict)
            viterbi_path.append(path_dict)

        best_state = None
        best_prob = -np.Inf
        for state in viterbi_prob[-1].keys():
            prob = np.log(self.transition_dict[state]['end'] / self.transition_counts[state])
            prob += viterbi_prob[-1][state]
            if best_prob < prob:
                best_prob = prob
                best_state = state

        states = [best_state]
        for path_dict in reversed(viterbi_path):
            states.append(path_dict[states[-1]])

        return list(reversed(states[:-1]))

    def confidence(self, sentence, answer):
        #call HMM viterbi.  return probabilities.  instead of returning the states, return the best prob.
        probs = []
        log_prob = 0
        state = 'start'
        for word, pos in zip(sentence,answer):
            emission_prob = self.emission_dict[pos][word] / self.emission_counts[pos]
            transition_prob = self.transition_dict[state][pos] / self.transition_counts[state]
            log_prob += np.log(emission_prob) + np.log(transition_prob)
            probs.append(emission_prob * transition_prob)
            state = pos
        transition_prob = self.transition_dict[state]['end'] / self.transition_counts[state]
        log_prob += np.log(transition_prob)
        return probs


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")

