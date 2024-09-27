#Steps for naiveBayes
#Load and clean the SMS data.
#Preprocess the text by converting it to lowercase, removing numbers, common words, and punctuation.
#Stem the words to reduce them to their base forms.
#Create a document-term matrix to represent the text numerically.
#Train a Naive Bayes model on the training data.
#Evaluate the model on the test data to see how well it classifies new messages.


#Step 1: read.csv(file.choose(), stringsAsFactors = FALSE): This loads the data from a CSV file that you select.
#The stringsAsFactors = FALSE means text data will be treated as plain text, not categories.
#str(sms_raw): Shows the structure of the data, like the number of columns and types of data.
#View(sms_raw): Opens the data in a spreadsheet-like view.
#sms_raw$type <- factor(sms_raw$type): Converts the "type" column (which tells if a message is "spam" or "ham") into a factor, which is important for classification.
#table(sms_raw$type): Counts how many "spam" and "ham" messages are in the data.
sms_raw <- read.csv(file.choose(), stringsAsFactors = FALSE) 
str(sms_raw)
View(sms_raw)
sms_raw$type <- factor(sms_raw$type)
str(sms_raw$type)
table(sms_raw$type)

#Text Cleaning: We need to clean up the text data to make it easier for the computer to work with.
#iconv(sms_raw$text): Converts the text encoding to make sure the text is readable by R.
#Corpus(VectorSource(sms_corpus)): Creates a collection of text documents from the SMS messages.
#inspect(sms_corpus[1:2]): Shows the first two messages in the corpus  to ensure that the text preprocessing has been done correctly.
#lapply(sms_corpus[1:2], as.character): Converts the first two messages into plain text.

install.packages("tm")
library(tm)
sms_corpus<-iconv(sms_raw$text)
sms_corpus <- Corpus(VectorSource(sms_corpus))
print(sms_corpus)
inspect(sms_corpus[1:2])
as.character(sms_corpus[[1]]) #This line converts the first document in the corpus into a character string.
lapply(sms_corpus[1:2], as.character) # return a list where each element is a character string representing the text of the corresponding document in the subset.

#tm_map: Applies a function to every document in the corpus to perform text processing tasks.
#tolower: Converts all text to lowercase, so "Hello" and "hello" are treated the same.
#removeNumbers: Removes all numbers from the text.
#removeWords(stopwords()): Removes common words like "and," "the," etc., which don't add much meaning.
#removePunctuation: Removes punctuation marks.

sms_corpus_clean <- tm_map(sms_corpus,
                           content_transformer(tolower)) 
as.character(sms_corpus[[1]])
as.character(sms_corpus_clean[[1]])
sms_corpus_clean <- tm_map(sms_corpus_clean, removeNumbers)
sms_corpus_clean <- tm_map(sms_corpus_clean,
                           removeWords, stopwords())
sms_corpus_clean <- tm_map(sms_corpus_clean, removePunctuation)
sms_corpus_clean
inspect(sms_corpus_clean[1:5]) #to verify the content of the first five documents after they have been cleaned and transformed

#Stemming: Reduces words to their root form. For example, "learning" and "learns" both become "learn."
#Stemming is commonly used in text mining and natural language processing (NLP) to normalize words so that different forms of a word are treated as the same.
#wordStem: Demonstrates stemming with some example words.
#stemDocument: Applies stemming to the entire corpus.

install.packages("SnowballC") #The SnowballC package provides functions for text stemming, which is the process of reducing words to their root form. 
library(SnowballC)
wordStem(c("learn", "learned", "learning", "learns"))
sms_corpus_clean <- tm_map(sms_corpus_clean, stemDocument)


#Document-Term Matrix (DTM): A matrix where each row is a message, and each column is a word.
#The values in the matrix indicate how many times each word appears in each message.
#Splitting Data: The data is split into a training set (first 4169 messages) and a test set (the remaining messages).
#The training set is used to build the model, and the test set is used to evaluate it.
sms_corpus_clean <- tm_map(sms_corpus_clean, stripWhitespace)
sms_dtm <- DocumentTermMatrix(sms_corpus_clean)
sms_dtm_train <- sms_dtm[1:4169, ]
sms_dtm_test <- sms_dtm[4170:5559, ]
sms_train_labels <- sms_raw[1:4169, ]$type #This line extracts the labels for the training set from the sms_raw data. 
sms_test_labels <- sms_raw[4170:5559, ]$type


#Word Cloud: A visual representation of word frequency, where more frequent words appear larger.
# The wordcloud package provides functions for creating word clouds,
#which are visual representations of text data
#where the size of each word indicates its frequency or importance.
#This helps you see which words are common in the messages.
install.packages("wordcloud")
library(wordcloud)
wordcloud(sms_corpus_clean, min.freq = 50, 
          random.order = FALSE,colors=brewer.pal(5,"Dark2")) 
#min.freq = 50: This argument specifies that only words with a frequency of at least 50 will be included in the word cloud.
#Words that appear less frequently than this threshold will be excluded.
#random.order = FALSE: This argument specifies that the words should be displayed in order of their frequency, with the most frequent words appearing larger and more prominently.
#If set to TRUE, words would be placed randomly in the cloud.
#specifies the color palette for the word cloud.
#brewer.pal(5, "Dark2") generates a set of 5 colors from the "Dark2" palette provided by the RColorBrewer package.
#This palette provides a set of visually distinct and attractive colors.

#findFreqTerms: Finds words that appear in at least 5 messages.
#sms_freq_words: Stores these frequent words.
findFreqTerms(sms_dtm_train, 5)
sms_freq_words <- findFreqTerms(sms_dtm_train, 5)
str(sms_freq_words)
sms_dtm_freq_train<- sms_dtm_train[ , sms_freq_words]
sms_dtm_freq_test <- sms_dtm_test[ , sms_freq_words]

#convert_counts: A function that changes word counts to "Yes" (word is present) or "No" (word is absent).
#apply: Applies this function to each word in the training and test sets.
#MARGIN specifies which dimension of the matrix to operate on.
#MARGIN = 2 means you’re applying the function to columns.
#If MARGIN were 1, the function would be applied to rows instead.
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}
sms_train <- apply(sms_dtm_freq_train, MARGIN = 2,
                   convert_counts)
sms_test <- apply(sms_dtm_freq_test, MARGIN = 2,
                  convert_counts)

#The e1071 package provides a collection of functions for various statistical and machine learning techniques
#including support vector machines (SVMs), Naive Bayes classification, clustering, and more.
#Naive Bayes Classifier: A simple yet effective model that predicts whether a message is "spam" or "ham" based on the words it contains.
#sms_classifier: This is the model trained on the training data.
#sms_test_pred: These are the model's predictions on the test data.
install.packages("e1071")
library(e1071)
sms_classifier <- naiveBayes(sms_train, sms_train_labels)
sms_test_pred <- predict(sms_classifier, sms_test)


#CrossTable: Compares the model's predictions (sms_test_pred) with the actual labels (sms_test_labels).
#This table shows how many messages were correctly or incorrectly classified as "spam" or "ham."
install.packages("gmodels")
library(gmodels)
CrossTable(sms_test_pred, sms_test_labels,
           prop.chisq = FALSE, prop.t = FALSE,
           dnn = c('predicted', 'actual')) 
