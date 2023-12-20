import tweepy
import re
import pandas as pd
from flask_assets import Bundle,Environment
import math
import os
import glob
from flask import Flask, request, render_template,redirect,url_for,session,send_file


consumer_key = 'QN13qkLriHk9HddSkEteFFDUX'
consumer_secret = 'F9LOAQHqReVfKXx4oM6wwT7dQTC1wL86J7bVcZ3pRv7XLMDlzt'

api_key = '1332225307617607681-eYcMkOY8x4TPYToluCIZGyGcKqHMQw'
api_secret = 'FVGuMsYOSiXyJg2vZLv75KsFTP8OvhitU7lineQFqTSqB'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret)
auth.set_access_token(api_key,api_secret)
#api = tweepy.API(auth)
api = tweepy.API(auth, wait_on_rate_limit=True)

app = Flask(__name__)
app.secret_key = 'Devesh'


js = Bundle('style3.js',output='main.js')
assets = Environment(app)
assets.register('main_js',js)
@app.route('/',methods=["GET","POST"])
def home():    
    return  render_template('dash.html')


files = glob.glob("static/images/*")
for f in files:
    os.remove(f)


def cleansing(text):
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r' ', text)
    text = re.sub(r'@[A-Za-z0-9]+',' ',text)
    text = re.sub(r'#',' ',text)
    text = re.sub(r'RT[\s]+',' ',text)
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
    no_punct = ""
    for char in text:
       if char not in punctuations:
           no_punct = no_punct + char
    text = no_punct
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|'\
                       '(?:%[0-9a-fA-F][0-9a-fA-F]))+',' ', text)
    
    
    return text


from textblob import TextBlob
def getsubjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def getpolarity(text):
    return TextBlob(text).sentiment.polarity

def getAnalysis(score):
    if score<0:
        return'Negative'
    elif score == 0.0:
        return'Neutral'
    elif score>0.1:
        return'Positive'
        
        
        
@app.route('/predict',methods=['GET','POST'])
def predict():
    #global df,name
    if request.method == "POST":     
        name = request.form.get('fname')
        ntweets = request.form['ntweets']
        #print(name)
        try:
            searched_tweets = [status for status in tweepy.Cursor(api.search, q=name,lang='en').items(int(ntweets))]
        except tweepy.TweepError:
            return render_template('error.html',error= 'Twitter API Error. Sorry, Could not fetch this much tweets, Please Retry Again with less number of tweets')
        df = pd.DataFrame([i.text for i in searched_tweets],columns = ['Tweets'])     
        temp = f1(df)
        before_nlp = temp[0]
        after_nlp = temp[1]
        RFC = temp[2]
        DTC = temp[3]
        LR = temp[4]
        SVC = temp[5]
        wordcloudtrain = temp[6]
        wordcloudtest = temp[7]
        nlp = temp[8]
        textblob = temp[9]
        beforenlppie = temp[10]
        afternlppie= temp[11]
        ptweets= temp[12]
        negtweets = temp[13]
        neutweets = temp[14]
        return render_template('result.html',before_nlp = after_nlp,after_nlp = before_nlp,
                               wordcloudtrain=('static/images/'+ wordcloudtrain+'.jpg'),wordcloudtest='/static/images/'+wordcloudtest+'.jpg',name = name,
                               tblob = '/static/images/'+textblob+'.jpg',nlp = ('static/images/'+nlp+'.jpg'),
                               pietrain='/static/images/'+beforenlppie+'.jpg',pietest = '/static/images/'+afternlppie+'.jpg',
                               RFC = RFC, DTC = DTC, LR = LR, SVC = SVC,ptweets = ptweets,negtweets=negtweets,
                               neutweets = neutweets)  

@app.route('/download',methods=['GET'])
def download_file():
    a = session.get('tweetsloc')
    return send_file('static/images/tweets'+a+'.csv',as_attachment=True)


@app.route('/aboutus.html',methods=['GET','POST'])
def aboutus():
    return render_template('aboutus.html')
@app.route('/dash.html',methods=['GET','POST'])
def dash():
    return render_template('dash.html')
'''@app.route('/all_tweets',methods=['GET','POST'])
def all_tweets():
    ptweets = request.args.get('ptweets', None)
    negtweets = request.args.get('negtweets', None)
    neutweets = request.args.get('neutweets', None)
    ptweets = session['ptweets']
    negtweets = session['negtweets']
    neutweets = session['neutweets']
    
    import json
    with open('ptweets.txt', 'r') as f:
        ptweets = json.loads(f.read())
    with open('negtweets.txt', 'r') as f:
        negtweets = json.loads(f.read())
    with open('neutweets.txt', 'r') as f:
        neutweets = json.loads(f.read())
    return render_template('column.html',ptweets=ptweets,negtweets=negtweets,
                          neutweets=neutweets) '''
def f1(df):
    import re
    import pandas as pd
    import nltk
   
    
        
    dflen = len(df)
    df['Tweets'] = df['Tweets'].apply(cleansing)
    df['Subjectivity'] = df['Tweets'].apply(getsubjectivity)
    df['Polarity'] = df['Tweets'].apply(getpolarity)
    df['Analysis'] = df['Polarity'].apply(getAnalysis)     
    
    
    ptweets,negtweets,neutweets = [],[],[]
    lendf  = len(df)
    pcnt,negcnt,neucnt = 0,0,0
    for i in range(lendf-1):
        if(pcnt==20 and negcnt==20 and neucnt==20):
            break
        elif(df['Analysis'][i]=='Positive'):
            ptweets.append(df['Tweets'][i])
            pcnt+=1
        elif(df['Analysis'][i]=='Negative'):
            negtweets.append(df['Tweets'][i])
            negcnt+=1
        else:
            neutweets.append(df['Tweets'][i])
            neucnt+=1              

    '''import json
    with open('static/ptweets.txt', 'w') as f:
        f.write(json.dumps(ptweets))   
    with open('static/negtweets.txt', 'w') as f:
        f.write(json.dumps(negtweets))
    with open('static/neutweets.txt', 'w') as f:
        f.write(json.dumps(neutweets))'''
    dfTweets = pd.DataFrame(list(zip(ptweets,negtweets,neutweets)),
              columns=['Positive','Negative', 'Neutral'])
    import time
    tweetsloc = str(time.strftime("%S", time.localtime())) 
    session['tweetsloc'] = tweetsloc
    dfTweets.to_csv('static/images/tweets'+tweetsloc+'.csv')
    
    '''session['ptweets'] = ptweets[:20]
    session['negtweets'] = negtweets[:20]
    session['neutweets'] = neutweets[:20]'''
    
    
    
    nltk.download('stopwords')
    
    from nltk.corpus import stopwords
    from nltk.stem.porter import PorterStemmer
    corpus = []
    for i in range(dflen):
        review = re.sub('[^a-zA-Z]',' ',df['Tweets'][i])
        review = review.lower()
        review = review.split()
        ps = PorterStemmer()
        all_stopwords = stopwords.words('english')
        all_stopwords.remove('not')
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)    
    
    corpus = pd.DataFrame(corpus,columns=['Tweets'])
    
    
    import math
    df_train_len = math.ceil(0.8 * len(corpus))
    df_train = corpus[:df_train_len]
    df_test = corpus[df_train_len:]
    
    
    
    #tokenization for training set
    df_train_data = []
    for i in range(df_train_len):   
        df_train_data.append(nltk.word_tokenize(corpus['Tweets'][i]))
    
    #tokenization for test set
    df_test_data = []
    for i in range(df_train_len,dflen):
        df_test_data.append(nltk.word_tokenize(corpus['Tweets'][i]))
    
    #Cleaning Part for Train Data
    df_train_data_len = len(df_train_data)
    for i in range(df_train_data_len):
            for k in range(len(df_train_data[i])):
                if(df_train_data[i][k]=='”' or df_train_data[i][k]=='“' or 'http' in df_train_data[i][k] or 
                   df_train_data[i][k]=='’' or df_train_data[i][k]== '—'  ):                
                    df_train_data[i][k] = ''
                     
                elif(df_train_data[i][k].isdigit()):                
                    df_train_data[i][k] = ''
                
        
    
    #Cleaning Part for Test Data
    df_test_data_len = len(df_test_data)
    for i in range(df_test_data_len):    
            for k in range(len(df_test_data[i])-1):
                if(df_test_data[i][k]=='”' or df_test_data[i][k]=='“' or 'http' in df_test_data[i][k] or 
                   df_test_data[i][k]=='’' or df_test_data[i][k]== '—'  ):
                     
                     df_test_data[i][k] = ''
                     
                elif(df_test_data[i][k].isdigit()):                
                    df_test_data[i][k] = ''
        
    
    #NORMALIZATION
    #lemmatization for training set
    from textblob import Word
    df_train_data_len = len(df_train_data)
    for i in range(df_train_data_len):
        for j in range(len(df_train_data[i])):
            w = Word(df_train_data[i][j])
            df_train_data[i][j] = w.lemmatize()
        
        
    #lemmatization for test set
    df_test_data_len = len(df_test_data)
    for i in range(df_test_data_len):
        for j in range(len(df_test_data[i])):
            w = Word(df_test_data[i][j])
            df_test_data[i][j] = w.lemmatize()
        
        
    #frequency distribution of training set
    train_words_freq = ''
    test_words_freq = ''
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt 
    for i in range(df_train_data_len):
        for j in range(len(df_train_data[i])): 
            tokens = df_train_data[i]
            train_words_freq += " ".join(tokens)+" "
    
    for i in range(df_test_data_len):
        for j in range(len(df_test_data[i])):
            tokens = df_train_data[i]
            test_words_freq += " ".join(tokens)+" "
    stopwords = set(STOPWORDS)      
    wordcloud_train = WordCloud(width = 800, height = 800, 
                    background_color ='black', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(train_words_freq) 
    import time
    # plot the WordCloud image                        
    plt.figure(facecolor = None) 
    plt.imshow(wordcloud_train) 
    plt.axis("off") 
    plt.tight_layout(pad = 0)  
      
    wordcloudtrain = 'wordcloudtrain'+ str(time.strftime("%S", time.localtime()))
 
    plt.savefig('static/images/'+ wordcloudtrain+'.jpg')
    plt.show() 
    
    
    wordcloud_test = WordCloud(width = 800, height = 800, 
                    background_color ='black', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(test_words_freq) 
    
    plt.figure(figsize = (6, 4), facecolor = None) 
    plt.imshow(wordcloud_test) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    
    wordcloudtest = 'wordcloudtest'+str(time.strftime("%S", time.localtime()))
    plt.savefig('static/images/'+ wordcloudtest+'.jpg')
    plt.show() 
    
    #joining the list
    for i in range(df_train_data_len):
        df_train_data[i] = ' '.join(df_train_data[i])
    
    for i in range(df_test_data_len):
        df_test_data[i] = ' '.join(df_test_data[i])
    
    
    
    train_test = df_train_data + df_test_data
    df_nlp = pd.DataFrame([i for i in train_test],columns = ['Tweets'])
    df_nlp['Subjectivity'] = df_nlp['Tweets'].apply(getsubjectivity)
    df_nlp['Polarity'] = df_nlp['Tweets'].apply(getpolarity)
    df_nlp['Analysis'] = df_nlp['Polarity'].apply(getAnalysis)
    
    # creating bag of words for training set
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer()
    x = cv.fit_transform(df_train_data).toarray()
    y = df_nlp['Polarity'][:df_train_data_len]
    #print(x.shape)
    #print(y.shape)
    import math
    for i in range(len(y)):
        if(y[i]<0.5):
            y[i] = math.floor(y[i])
        else:
            y[i] = math.ceil(y[i])
            
    # creating bag of words for test set
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    tf = TfidfVectorizer()
    cv = CountVectorizer()
    x_test = cv.fit_transform(df_test_data).toarray()
    
    
    
    # splitting the training data into train and valid sets
    
    from sklearn.model_selection import train_test_split
    
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size = 0.25,random_state=0)
    
    
    # standardization
    
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    from sklearn import preprocessing
    
    
    lab_enc = preprocessing.LabelEncoder()
    x_train = sc.fit_transform(x_train)
    x_valid = sc.transform(x_valid)
    x_test = sc.fit_transform(x_test)
    x_test = sc.transform(x_test)
    
    
    
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import f1_score
    
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    
    #print('\n\n Random Forest Classifier')
    y_pred = model.predict(x_valid)
    pd.Series(y_pred).value_counts().plot.bar(color = 'pink', figsize = (6, 4))
    
    #print("Training Accuracy :", model.score(x_train, y_train))
    #print("Validation Accuracy :", model.score(x_valid, y_valid))
    RFC_score = model.score(x_valid, y_valid)
    # calculating the f1 score for the validation set
    #print("F1 score :", f1_score(y_valid, y_pred,average= 'micro'))
    
    # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    #print(cm)
    
    
    from sklearn.tree import DecisionTreeClassifier
    
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    
    #print('\n\n Decision Tree Classifier')
    y_pred = model.predict(x_valid)
    pd.Series(y_pred).value_counts().plot.bar(color = 'pink', figsize = (6, 4))
    
    #print("Training Accuracy :", model.score(x_train, y_train))
    #print("Validation Accuracy :", model.score(x_valid, y_valid))
    DTC_score = model.score(x_valid, y_valid)
    
    
    # calculating the f1 score for the validation set
    #print("F1 score :", f1_score(y_valid, y_pred,average= 'micro'))
    
    # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    #print(cm)
    LR_score = 0
    
    from sklearn.linear_model import LogisticRegression
        
    model = LogisticRegression()
    model.fit(x_train, y_train)
        
    #print('\n\n Logistic Regression')
    y_pred = model.predict(x_valid)
    pd.Series(y_pred).value_counts().plot.bar(color = 'pink', figsize = (6, 4))
        
    #print("Training Accuracy :", model.score(x_train, y_train))
    #print("Validation Accuracy :", model.score(x_valid, y_valid))
    LR_score = model.score(x_valid, y_valid)
        
        # calculating the f1 score for the validation set
    #print("F1 score :", f1_score(y_valid, y_pred,average= 'micro'))
        
        # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    #print(cm)
    
    SVC_score = 0
   
    from sklearn.svm import SVC
        
    model = SVC()
    model.fit(x_train, y_train)
        
    #print('\n\n Support Vector Classifier')
    y_pred = model.predict(x_valid)
    pd.Series(y_pred).value_counts().plot.bar(color = 'pink', figsize = (6, 4))
        
    #print("Training Accuracy :", model.score(x_train, y_train))
    #print("Validation Accuracy :", model.score(x_valid, y_valid))
    SVC_score = model.score(x_valid, y_valid)
        
        # calculating the f1 score for the validation set
    #print("F1 score :", f1_score(y_valid, y_pred,average= 'micro'))
        
        # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    #print(cm)
    #except ValueError:pass
    
    '''from sklearn.naive_bayes import GaussianNB 
    
    model = GaussianNB() 
    model.fit(x_train, y_train)
    
    
    print('\n\n Naive Bayes Classifier')
    y_pred = model.predict(x_valid)
    pd.Series(y_pred).value_counts().plot.bar(color = 'pink', figsize = (6, 4))
    
    print("Training Accuracy :", model.score(x_train, y_train))
    print("Validation Accuracy :", model.score(x_valid, y_valid))
    
    # calculating the f1 score for the validation set
    print("F1 score :", f1_score(y_valid, y_pred,average= 'micro'))
    
    # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    print(cm)
    
    
    from sklearn.neighbors import KNeighborsClassifier
    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    
    print('\n\n KNeigbors Classifier')
    y_pred = model.predict(x_valid)
    pd.Series(y_pred).value_counts().plot.bar(color = 'pink', figsize = (6, 4))
    
    print("Training Accuracy :", model.score(x_train, y_train))
    print("Validation Accuracy :", model.score(x_valid, y_valid))
    
    # calculating the f1 score for the validation set
    print("F1 score :", f1_score(y_valid, y_pred,average= 'micro'))
    
    # confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    print(cm)'''
    
    
    
    #result with text blob analysis
    #pd.Series(df['Analysis']).value_counts().plot.bar(color = 'blue').get_figure().savefig('static/textblob.jpg',bbox_inches='tight')
    #results by classifiers
    #pd.Series(df_nlp['Analysis']).value_counts().plot.bar(color = 'maroon').get_figure().savefig('static/nlp.jpg',bbox_inches='tight')
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    senti = ['Positive','Negative','Neutral']
    tw = [int(list(df['Analysis']).count('Positive')),
                  int(list(df['Analysis']).count('Negative')),
                  int(list(df['Analysis']).count('Neutral'))]
    plt.xlabel("Sentiments")
    plt.ylabel("No. of Total Tweets")
    plt.title("Before NLP")
    ax.bar(senti,tw,width=0.5,color='blue')
    
    nlp = 'nlp' + str(time.strftime("%S", time.localtime()))
    plt.savefig('static/images/'+nlp+'.jpg', bbox_inches='tight')
    plt.show()      
    
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    senti = ['Positive','Negative','Neutral']
    tw = [int(list(df_nlp['Analysis']).count('Positive')),
                  int(list(df_nlp['Analysis']).count('Negative')),
                  int(list(df_nlp['Analysis']).count('Neutral'))]
    plt.xlabel("Sentiments")
    plt.ylabel("No. of Total Tweets")
    plt.title("After NLP")
    ax.bar(senti,tw,width=0.5,color='maroon')
   
    textblob = 'textblob' +str(time.strftime("%S", time.localtime()))
    plt.savefig('static/images/'+textblob+'.jpg', bbox_inches='tight')
    plt.show() 
    
    '''import matplotlib.pyplot as plt
    plt.scatter(y_valid, y_pred)
    plt.xlabel('True Values ')
    plt.ylabel('Predictions ')
    plt.show()'''
    
    import numpy as np
    
    pi = np.array([int(list(df['Analysis']).count('Positive')),
                  int(list(df['Analysis']).count('Negative')),
                  int(list(df['Analysis']).count('Neutral'))])
    
    
    pi_nlp = np.array([int(list(df_nlp['Analysis']).count('Positive')),
                  int(list(df_nlp['Analysis']).count('Negative')),
                  int(list(df_nlp['Analysis']).count('Neutral'))])
    
    
    plt.pie(pi, labels = ["Positive", "Negative", "Neutral"],shadow=True)
    plt.title('Before NLP')  
    plt.axis("off")    
    beforenlppie = 'beforenlppie'+str(time.strftime("%S", time.localtime()))
    plt.savefig('static/images/'+beforenlppie+'.jpg')
    plt.show()
    
    plt.pie(pi_nlp, labels = ["Positive", "Negative", "Neutral"],shadow=True)
    plt.title('After NLP')
    plt.axis("off")    
    afternlppie = 'afternlppie' +str(time.strftime("%S", time.localtime()))
    plt.savefig('static/images/'+afternlppie+'.jpg')
    plt.show()
    
    before_nlp = (' \nPositive:'+ str(list(df['Analysis']).count('Positive'))+'\n'+
          '\nNegative:'+str(list(df['Analysis']).count('Negative'))+'\n'+
          '\nNeutral:'+ str(list(df['Analysis']).count('Neutral')))
    
    
    after_nlp = ('\nPositive:'+ str(list(df_nlp['Analysis']).count('Positive'))+'\n'+
          '\nNegative:'+str(list(df_nlp['Analysis']).count('Negative'))+'\n'+
          '\nNeutral:'+ str(list(df_nlp['Analysis']).count('Neutral')))
    
    return [before_nlp, after_nlp,RFC_score,DTC_score,LR_score,SVC_score,wordcloudtrain,wordcloudtest,nlp,textblob,beforenlppie,afternlppie,ptweets,negtweets,neutweets]


'''if __name__ == "__main__":     
     app.run(host = '0.0.0.0',port=8080)'''

if __name__ == "__main__":     
     app.run(debug=True, use_reloader=False)