# importing necessary packages
import nltk                                 
import io                                                       
import sys          
import os                                            
import pandas as pd                                             
from nltk.corpus import stopwords                               
from nltk.tokenize import word_tokenize                         
from sklearn.feature_extraction.text import TfidfVectorizer  
from nltk.stem import PorterStemmer   
from sklearn.cluster import KMeans                              
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet

# reading the file to dataframe
def read_from_excel(filename):
	df = pd.read_excel(filename)
	if(filename == "Type5.xlsx"):
		df = df[df.Question.notnull()]
		df = df.reset_index(drop=True)
	if(filename == "SampleOutput.xlsx"):
		df = df[["Question","Answers"]]
		df = df.dropna(how="all",axis=0)
		df = df.reset_index(drop=True)
	return df

# writing the dataframe to excel
def write_to_excel(df,filename):
	writer = pd.ExcelWriter(filename)
	df.to_excel(writer,"Sheet1")
	writer.save()

# clustering the data based on tf-idf vectors using k-means clustering
def cluster_the_data(df,documents_list):
	question_freq = [0]*df.shape[0]
	df["question_freq"] = question_freq
	tfidf_matrix = vectorizer.fit_transform(documents_list) # converting the question list to the tf-idf matrix
	kmeans.fit(tfidf_matrix.toarray()) # fitting the matrix to k-means clustering  
	return (kmeans,tfidf_matrix)

# gives the suggestions for the user queries
def actual_process(query,df,auto):
	shape = vectorizer.transform([query]).shape
	treshold = 0 # modifying basing on our convenience
	nearest_centroids = []
	for ithcentroid in range(len(centroids)):
		# finding cosine similairty between the user_query and every centroid
		similarity = cosine_similarity(centroids[ithcentroid].reshape(shape),vectorizer.transform([query])) 
		# noting the centriods basing on the threshold
		if(similarity>treshold):
			nearest_centroids.append([similarity[0][0],ithcentroid])  
	nearest_centroids = sorted(nearest_centroids, key=lambda item: item[0], reverse = True)

	# getting the indexes (points) with in the clusters belonging to the nearest centroids  
	indexes = []
	for i in range(len(nearest_centroids)):
		k = list(df["centroids"].values == nearest_centroids[i][1])
		for i in range(len(k)):
			if(k[i]==True):
				indexes.append(i)

	# finding the cosine similarities between the indexes and queries, sorting them.
	cosine_similarity_q = []
	for row in range(len(indexes)):
		c_similarity = cosine_similarity(tfidf_matrix[indexes[row]],vectorizer.transform([query]))
		if(c_similarity > treshold):
			cosine_similarity_q.append([c_similarity[0][0],indexes[row]])
	cosine_similarity_q = sorted(cosine_similarity_q, key=lambda item: item[0], reverse = True)

	# predicting and suggesting based on the previously sorted values 
	for q in range(len(cosine_similarity_q)):
		row_value = cosine_similarity_q[q][1]
		print ()
		print("Suggested Question=",df["Question"][row_value])
		print("Suggested Answer= ",df["Answer"][row_value])
		print ()
		if(auto==0):
			print ("****** next  ******")
			break
		if(auto == 1):
			print ("****** choose  ******")
			print ("press 1 if you are satisfied with the answer")
			print ("press any other number for next suggestion")

			# verifing with the user for wich question he has satisfied 
			flag_question = int(input())
			if(flag_question == 1):
				# appending the user queries if user satisfies with the predictions
				df.at[row_value,"user_questions"] = df.at[row_value,"user_questions"]+" "+query
				df.at[row_value,"question_freq"] = int(df.at[row_value,"question_freq"])+1
				break

	sum_of_user_questions = df["question_freq"].sum()

	# reclustering on the new data (got from the addition of previously asked questions) after certain re_cluster value
	recluster_value = 4 
	if(sum_of_user_questions >= recluster_value):
		write_to_excel(df,"Type5.xlsx")
		print("your questions are noted down. for updation of system we want you to re-run the application. Thank you")
		sys.exit(0)

# preprocessing the query by first tokenizing, removing stopwords, stemming and then converting it to lowercase
def find_answer(query,auto):
	query = " ".join(ps.stem(word.lower()) for word in word_tokenize(query) if word not in stop_words and (len(wordnet.synsets(word))>0))
	actual_process(query,df,auto)

df = read_from_excel("Type5.xlsx")

vectorizer = TfidfVectorizer(min_df=1) 
kmeans = KMeans(n_clusters = 10) #initialising k-means with 10 clusters
stop_words = stopwords.words('english')
ps = PorterStemmer()
len_of_df = df.shape[0]

if(df.shape[1]==2):
	user_questions = ["."]*len_of_df
	df["user_questions"] = user_questions


questions_list= list(df["Question"])
answers_list = list(df["Answer"])
user_questions = list(df["user_questions"])

#preprocessing the questions by first tokenizing, removing stopwords, stemming and then converting it to lowercase  
documents_list = []
for i in range(len_of_df):
	noStopWords = " ".join(ps.stem(word.lower()) for word in word_tokenize(questions_list[i]+" "+user_questions[i]) if word not in stop_words and (len(wordnet.synsets(word))>0))
	documents_list.append(noStopWords)

# assgning values to the centroids returned from the clustering function 
kmeans = cluster_the_data(df,documents_list)[0]
tfidf_matrix = cluster_the_data(df,documents_list)[1]
centroids = kmeans.cluster_centers_
df["centroids"] = kmeans.labels_     # appending the centroids to dataframee
 
# function for testing the SampleOutput (predictions)
def test():
	df_test = read_from_excel("SampleOutput.xlsx")

	for i in range(df_test.shape[0]):
		print("actual_question= ",df_test["Question"][i])
		print("actual_answer= ",df_test["Answers"][i])
		find_answer(df_test["Question"][i],auto=0)

#main function for asking query 
def main(argv):
	
	if(argv=="prediction"):
		test()

	else:	
		while (True):
			print ("enter your question: ")
			user_query = input()
			find_answer(user_query,auto=1)

			print ("type 'EXIT' to exit or press any number to continue asking questions")
			exit = str(input())
			print (exit)
			if(exit == "EXIT"):
				break
			else:
				os.system( 'cls' )


if __name__ == '__main__':
	if(len(sys.argv)>1):
		main(sys.argv[1])
	else:
		main("")


