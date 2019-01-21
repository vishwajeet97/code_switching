#!/usr/bin/python
# -*- coding: utf-8 -*-

def language_word(word):
	"""
	word: a unicode string with no whitespaces
	Returns: 	0 if all characters are in English
				1 if all characters are in Chinese
				2 otherwise
	"""
	chines_counter = 0
	english_couner = 0
	for ch in word:
		if ch >= u'\u4e00' and ch <= u'\u9fff':
			chines_counter += 1
		else:
			english_couner += 1

	if chines_counter != 0 and english_couner != 0:
		return 2
	else:
		if chines_counter != 0:
			return 1
		else:
			return 0

def segment_string(string):
	"""
	string: a unicode string with whitespaces
	Returns: A list of list of unicode strings without spaces 
	"""
	segment_string = []
	segment_string.append([])
	for index, word in enumerate(string.split()):
		if len(segment_string[-1]) == 0:
			segment_string[-1].append(word)
		elif language_word(segment_string[-1][-1]) == language_word(word):
			segment_string[-1].append(word)
		else:
			segment_string.append([])
			segment_string[-1].append(word)

	return segment_string

import codecs

for data in ["train", "test", "dev"]:
	with codecs.open("data/SEAME/%s.txt"%data, encoding='utf-8') as fin:
		lines = fin.readlines()

	corpus = []
	total_words = 0
	english_words = 0
	chinese_words = 0
	chinese_dict = {}
	english_dict = {}
	total_unique = {}

	for line in lines:
		words = line.strip().split()
		for word in words:
			if language_word(word) == 0:
				english_words += 1
				if word not in english_dict.keys():
					english_dict[word] = 1
				else:
					english_dict[word] += 1
			elif language_word(word) == 1:
				chinese_words += 1
				if word not in chinese_dict.keys():
					chinese_dict[word] = 1
				else:
					chinese_dict[word] += 1

			if word not in total_unique.keys():
				total_unique[word] = 1
			else:
				total_unique[word] += 1

			total_words += 1

	print("-"*15 + ("%s set " % data) + "-"*15)
	print("Total number of words : %d" % total_words)
	percent_zh = (chinese_words/total_words)*100
	percent_eng = (english_words/total_words)*100
	print("The percentage of Zh words : %f" % percent_zh)
	print("The percentage of Eng words : %f" % percent_eng)
	print("The number of unique Zh words : %d" % len(chinese_dict.keys()))
	print("The number of unique Eng words : %d" % len(english_dict.keys()))
	print("The sum is : %d" % (len(chinese_dict.keys()) + len(english_dict.keys())))
	print("The number of unique tokens : %d" % len(total_unique.keys()))

	# lines = [segment_string(line.strip()) for line in lines]

	# print(lines[0])

	# with open("data/SEAME/segmentation_%s.txt" % data, 'w') as fout:
	# 	for sentence in lines:
	# 		output = ''
	# 		for index, segment in enumerate(sentence):
	# 			if index != len(sentence)-1:
	# 				output += ("0 "*(len(segment)-1)) + ("1 ")
	# 			else:
	# 				output += ("0 "*len(segment))

	# 		fout.write(output+"\n")
