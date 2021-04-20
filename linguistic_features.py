import pandas as pd
import string
import re
import nltk
from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')



data = pd.read_csv("news_headlines.csv")

r1_lengths_dict = {}
r1_freq_dict = {}
r1_punct_dict = {}

r2_lengths_dict = {}
r2_freq_dict = {}
r2_punct_dict = {}

r3_lengths_dict = {}
r3_freq_dict = {}
r3_punct_dict = {}

r1_http = 0
r2_http = 0
r3_http = 0

punctuation = string.punctuation + "’"

for i in range(data.shape[0]):
	up_votes = data.iloc[i, 0]
	title = data.iloc[i,1]
	title_length = len(title.split(" "))

	if up_votes == "0-100":
		if ".com" in title or "www." in title:
			r1_http += 1

		if title_length in r1_lengths_dict:
			r1_lengths_dict[title_length] += 1
		else:
			r1_lengths_dict[title_length] = 1

		tokens = word_tokenize(title)
		for word in tokens:
			if word in punctuation:
				if word in r1_punct_dict:
					r1_punct_dict[word] += 1
				else:
					r1_punct_dict[word] = 1
				continue
			word = word.lower()
			w = wnl.lemmatize(word)
			if w not in english_stopwords:
				if word in r1_freq_dict:
					r1_freq_dict[word] += 1
				else:
					r1_freq_dict[word] = 1


	elif up_votes == "100-1000":
		if ".com" in title or "www." in title:
			r2_http += 1

		if title_length in r2_lengths_dict:
			r2_lengths_dict[title_length] += 1
		else:
			r2_lengths_dict[title_length] = 1

		tokens = word_tokenize(title)
		for word in tokens:
			if word in punctuation:
				if word in r2_punct_dict:
					r2_punct_dict[word] += 1
				else:
					r2_punct_dict[word] = 1
				continue
			word = word.lower()
			w = wnl.lemmatize(word)
			if w not in english_stopwords:
				if word in r2_freq_dict:
					r2_freq_dict[word] += 1
				else:
					r2_freq_dict[word] = 1

	elif up_votes == ">1000":
		if ".com" in title or "www." in title:
			r3_http += 1

		if title_length in r3_lengths_dict:
			r3_lengths_dict[title_length] += 1
		else:
			r3_lengths_dict[title_length] = 1

		tokens = word_tokenize(title)
		for word in tokens:
			if word in punctuation:
				if word in r3_punct_dict:
					r3_punct_dict[word] += 1
				else:
					r3_punct_dict[word] = 1
				continue
			word = word.lower()
			w = wnl.lemmatize(word)
			if w not in english_stopwords:
				if word in r3_freq_dict:
					r3_freq_dict[word] += 1
				else:
					r3_freq_dict[word] = 1

print("---- length of titles ----")
print("upvotes: 0-100")
sorted_r1_lengths_dict = sorted(r1_lengths_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r1_lengths_dict[0:10])

print("upvotes: 100-1000")
sorted_r2_lengths_dict = sorted(r2_lengths_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r2_lengths_dict[0:10])

print("upvotes: 1000+")
sorted_r3_lengths_dict = sorted(r3_lengths_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r3_lengths_dict[0:10])

print()
print()
print("---- top 20 most frequent words ----")
print("upvotes: 0-100")
sorted_r1_word_dict = sorted(r1_freq_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r1_word_dict[0:20])

print("upvotes: 100-1000")
sorted_r2_word_dict = sorted(r2_freq_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r2_word_dict[0:20])

print("upvotes: 1000+")
sorted_r3_word_dict = sorted(r3_freq_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r3_word_dict[0:20])

print()
print()
print("---- most frequent punctuation ----")
print("upvotes: 0-100")
sorted_r1_punct_dict = sorted(r1_punct_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r1_punct_dict)

print("upvotes: 100-1000")
sorted_r2_punct_dict = sorted(r2_punct_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r2_punct_dict)

print("upvotes: 1000+")
sorted_r3_punct_dict = sorted(r3_punct_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_r3_punct_dict[0:20])

print()
print()
print("---- total number of web links ----")
print("upvotes: 0-100")
print(r1_http)
print("upvotes: 100-1000")
print(r2_http)
print("upvotes: 1000+")
print(r3_http)

'''

# title length

titles = d['title']
title_lengths = [len(title.split(" ")) for title in titles]
length_set = set(title_lengths)

lengths_dict = {}
for length in length_set:
	lengths_dict[length] = title_lengths.count(length)

sorted_lengths_dict = sorted(lengths_dict.items(), key=lambda x:x[1], reverse=True)
#print(sorted_lengths_dict)

# numbers of links in title

punct_dict = {}
for title in titles:
	result = re.findall(r'[^\w\s]', title)
	freq = len(result) + title.count("_") + title.count("’") 

	if freq in punct_dict:
		punct_dict[freq] += 1
	else:
		punct_dict[freq] = 1

sorted_punct_dict = sorted(punct_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_punct_dict)

# most frequent words in title

punctuation = string.punctuation + "’"

word_dict = {}
for title in titles:
	tokens = word_tokenize(title)
	for word in tokens:
		word = word.lower()
		if word in punctuation:
			continue
		w = wnl.lemmatize(word)
		if w not in english_stopwords:
			if word in word_dict:
				word_dict[word] += 1
			else:
				word_dict[word] = 1

sorted_word_dict = sorted(word_dict.items(), key=lambda x:x[1], reverse=True)
print(sorted_word_dict[0:20])
'''
