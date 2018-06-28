# FaqFinder
A smart system  to find a query in a list of Faq's

### requirements
python 3.x
pip install sklearn                         --- sklearn framework
pip install numpy                           --- python numpy package
pip install pandas                          --- python pandas package
nltk.download('stopwords')                  --- nltk module consisting of stopwords
nltk.download('punkt')                      --- nltk tokenizer
** Recomended Anaconda python 3.x distribution

### Algorithm 
step-1 K-means clustering on question in the input file. using tf-idf vectorization
step-2 find the cosine distance of question, then comparing it with cluster centroids and select the nearest cluster centroid to the question
step-3 calculate the distance of question with every point(question) in the nearest cluster.
step-4 output the minimum three distances Or three most relavent questions.
step-5 store the user questions to the dataframe for future training. Append the new questions with the old questions and Recluster by keeping certain threshold.
step-6 save the new dataframe to excel file so as to not lose the data.

### demo link 

DEMO URL1: https://drive.google.com/open?id=1Ygdp3wcjW7IhZoqKyiU-HAG3_w1wQqKh
DEMO URL2: https://drive.google.com/open?id=1BC5M4kevSlCpsXm1fvnXM7pUE5-lGlKC

