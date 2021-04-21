My research question for this DS/ML challenge was, “Is the title of a news article enough to predict how many upvotes the article would receive?”

I approached this problem in two parts. First, I divided the titles based on the number of upvotes (0-100, 100-1000, and 1000+). Each range has 14546 titles. This was displayed in a csv file named news_headlines.csv, created by running create_csv.py. I then observed the linguistic features of the titles: title length, most frequent words, punctuation, and URL links. The code can be found in linguistic_features.py.

From the output in linguistic_analysis_output, we found that there is very little variation found in the titles across all upvote categories. The typical title lengths are in the range of 8-10 words, and the most frequent words and punctuation are very similar. It is interesting to note that titles with more than 1000 upvotes tend to contain the least amount of URL links in its titles, but this is not by a significant amount.

The second task is text classification. I trained Multinomial Naïve Bayes, Random Forest, and Support Vector Machine classifiers to predict the number of upvotes an article will receive based on the Tf-idf features of the titles. After fine-tuning and further preprocessing, the accuracies of the classifiers are listed below:

-------------------
Naive Bayes, accuracy = 0.5104
-------------------
Random Forest, accuracy = 0.5216
-------------------
Linear SVM, accuracy = 0.5299
---------------------

Clearly, the accuracies indicate that classifiers trained on features of the titles cannot make good predictions. I decided to make an additional classification on the emotion-based words from the NRC Emotion corpus. Popular articles tend to rely on fearmongering and sensationalist titles to incite readers to click on them so I predicted that articles with more upvotes will contain more emotional words.

I created 15-grams (the average length of a title found in news_headlines.csv) of each emotion category (anger, anticipation, disgust, fear, joy, sadness, surprise, trust, negative, positive) and then predicted whether the ngrams were more likely to receive 0-100, 100-1000, or 1000+ upvotes. The result of this, found in NRC_classification_output, is inconsistent, with most of the ngrams of all emotional categories predicted as receiving 0-100 upvotes in one run and most ngrams predicted as receiving 1000+ upvotes in another run. In both cases, the other classes of upvotes had about the same number of predicted ngrams. Only through use of stratified k-fold grouping did the results stopped swapping. Thus, we cannot accurately confirm that any range of upvotes contain titles that are more emotional than the others, as it depends on how the test and train data was split.

The classification can be performed by running the classifiers.sh file with the command line: ./classifiers.sh. Running time is around 4-5 hours. Other python files (such as those listed previously) can be run with the command line: python3 [python file name].

In conclusion, the results of the two tasks proves that the title alone is not enough to predict the number of upvotes an article would receive. A good title, in fact, should be focused on getting a reader to click on the news article, and based on the quality of the article itself, the reader then chooses whether or not to upvote. Thus, as future work, I would like to perform the same classification on the text of the articles to see if it is a better indicator of upvotes. It would also be interesting to see the results across different news categories, such as sports, entertainment, etc.
