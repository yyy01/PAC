import random
from random import shuffle
random.seed(0)

def random_swap(words, n):
	new_words = words.copy()
	for _ in range(n):
		new_words = swap_word(new_words)
	return new_words

def swap_word(new_words):
	random_idx_1 = random.randint(0, len(new_words)-1)
	random_idx_2 = random_idx_1
	counter = 0
	while random_idx_2 == random_idx_1:
		random_idx_2 = random.randint(0, len(new_words)-1)
		counter += 1
		if counter > 3:
			return new_words
	new_words[random_idx_1], new_words[random_idx_2] = new_words[random_idx_2], new_words[random_idx_1] 
	return new_words

def eda(sentence, alpha = 0.3, num_aug = 5):
	words = sentence.split(' ')
	num_words = len(words)
	augmented_sentences = []

	if (alpha > 0):
		n_rs = max(1, int(alpha*num_words))
		for _ in range(num_aug):
			a_words = random_swap(words, n_rs)
			augmented_sentences.append(' '.join(a_words))

	augmented_sentences = [sentence for sentence in augmented_sentences]
	shuffle(augmented_sentences)
	if num_aug >= 1:
		augmented_sentences = augmented_sentences[:num_aug]
	else:
		keep_prob = num_aug/len(augmented_sentences)
		augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]

	return augmented_sentences

if __name__ == '__main__' :
	sentence  = 'Grey seals have no ear flaps and their ears canals are filled with wax.Grey seals hear better underwater when their ears open like a valve.Dogs have sensitive ears that can hear as far as a quarter of a mile away.'
	print(eda(sentence))