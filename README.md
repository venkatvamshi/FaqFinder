# FaqFinder
A smart system  to find a query in a list of Faq's

### requirements
python 3.x
pip install openpyxl                        --- python package for writing data into excel file (.xlsx)
pip install xlrd                            --- python package for reading excel file (.xlsx)
pip install sklearn                         --- sklearn framework
pip install numpy                           --- python numpy package
pip install pandas                          --- python pandas package
nltk.download('stopwords')                  --- nltk module consisting of stopwords
nltk.download('punkt')                      --- nltk tokenizer
** Recomended Anaconda python 3.x distribution

### Algorithm 
step-1 K-means clustering on tags in the input file. using tf-idf vectorization
step-2 find the cosine distance of question from different cluster centroids and select the nearest cluster centroid to the question
step-3 calculate the distance of question from every point in the nearest cluster.
step-4 output the minimum three distances. Or three most relavent questions.
step-5 store the user questions for future training. Recluster using the old questions and the new questions at the end of the day.
step-6 save the new dataframe to excel file so as to not lose the data.

### demo link 

