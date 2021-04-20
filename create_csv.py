import csv
import random

csv_file = open("news_headlines.csv", "w+")

writer = csv.DictWriter(csv_file, fieldnames=["upvotes", "text"])
writer.writeheader()

data_file = open("Eluvio_DS_Challenge.csv").read()
data = data_file.splitlines()

r1_texts = [] # titles with 0-100 upvotes
r2_texts = [] # titles with 100-1000 upvotes
r3_texts = [] # titles with 1000+ upvotes
for news in data:
	if news.split(",")[0] == "time_created":
		continue
	
	# replace the delimiter comma with ";;;" to avoid splitting at the commas in the titles
	news_delim = news.replace('","', '";;;"')
	news_elem = news_delim.split('";;;"')

	if len(news_elem) != 8:
		continue

	text = news_elem[4]

	up_vote = int(news_elem[2])
	if up_vote > 0 and up_vote <= 100:
		r1_texts.append("r1:: " + text)
	elif up_vote <= 1000 and up_vote > 100:
		r2_texts.append("r2:: " + text)
	elif up_vote > 1000:
		r3_texts.append("r3:: " + text)


# the number of texts for all three ranges becomes the same
cut_off_len = len(r3_texts) 
r1_texts = r1_texts[:cut_off_len]
r2_texts = r2_texts[:cut_off_len]

all_texts = r1_texts + r2_texts + r3_texts
random.shuffle(all_texts)

for text in all_texts:
	if "r1:: " in text:
		text = text.replace("r1:: ", "")
		writer.writerow({"upvotes":"0-100", "text":text})
	elif "r2:: " in text:
		text = text.replace("r2:: ", "")
		writer.writerow({"upvotes":"100-1000", "text":text})
	elif "r3:: " in text:
		text = text.replace("r3:: ", "")
		writer.writerow({"upvotes":">1000", "text":text})


