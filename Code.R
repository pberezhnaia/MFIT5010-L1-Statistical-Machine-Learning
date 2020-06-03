library(readr)
library(qdapDictionaries)
library(qdapTools)
library(stopwords)
library(textstem)
library(caTools)
library(caret)
library(rword2vec)
library(textclean)
library(wordcloud)
library(tm)

options(scipen=999)
set.seed(22)

#data merging
files_neg<-c(list.files("/Users/polina_berezhnaia/Dropbox/HKUST/ML/Project/train/neg",
                        pattern="*.txt", full.names=TRUE),
             list.files("/Users/polina_berezhnaia/Dropbox/HKUST/ML/Project/test/neg",
                        pattern="*.txt", full.names=TRUE))
files_pos<-c(list.files("/Users/polina_berezhnaia/Dropbox/HKUST/ML/Project/train/pos",
                        pattern="*.txt", full.names=TRUE),
             list.files("/Users/polina_berezhnaia/Dropbox/HKUST/ML/Project/test/pos",
                        pattern="*.txt", full.names=TRUE))

df<-data.frame()
colnames(df)<-"text"
for(i in 1:length(files_neg)){
  df<-rbind(df, read_delim(files_neg[i],"\t",escape_double=FALSE,trim_ws=TRUE,
                           col_names=c("text")))}
neg<-as.data.frame(cbind(df[-1,], c(rep("negative",nrow(df)-1))))
colnames(neg)<-c("text","sentiment")

df<-data.frame()
colnames(df)<-"text"
for(i in 1:length(files_pos)){
  df<-rbind(df, read_delim(files_pos[i],"\t",escape_double=FALSE,trim_ws=TRUE,
                           col_names=c("text")))
}
pos<-as.data.frame(cbind(df[-1,], c(rep("positive",nrow(df)-1))))
colnames(pos)<-c("text","sentiment")

df<-rbind(neg,pos)
df$text<-as.character(df$text)

# text preprocessing
contractions<-contractions
add<-data.frame(cbind(gsub("'","",contractions$contraction),contractions$expanded),
                stringsAsFactors = FALSE)
colnames(add)<-colnames(contractions)
contractions<-rbind(contractions,add)
stop_words<-stopwords("en")
stop_words<-stop_words[-c(165,167)]

preprocess<-function(x){
  x<-gsub('@\\S+',"", x)
  x<-gsub('#\\S+',"", x)
  x<-replace_html(x)
  x<-gsub('[0-9]+',"", x)
  x<-gsub('[[:punct:]]',"", x)
  for(i in nrow(contractions)){x<-gsub(as.character(contractions[i,1]),
                                       paste(" ",as.character(contractions[i,2])," "),x)}
  x<-gsub("^[[:space:]]*","",x)
  x<-gsub("[[:space:]]*$","",x)
  x<-gsub(' +',' ',x)
  x<-tolower(x)
  x<-replace_word_elongation(x)
  x<-removeWords(x,c(stop_words,"im","ive"))
  x<-lemmatize_strings(x)
  x<-gsub('[0-9]+',"", x)
  return(x)
}

for(i in 1:nrow(df)){
  df[i,1]<-as.character(preprocess(df[i,1]))
}


#train and test
split<-sample.split(df$sentiment, SplitRatio=0.85)
train<-df[split,]
y_train<-train$sentiment
test<-df[!split,]
y_test<-test$sentiment
table(train$sentiment)

#corpus
train_corpus<-VCorpus(VectorSource(train$text))
test_corpus<-VCorpus(VectorSource(test$text))

#train
#unigrams
tfidf<-function(x){weightTfIdf(x,normalize=FALSE)}
terms<-DocumentTermMatrix(train_corpus,control=list(weighting=tfidf))
sparse<-removeSparseTerms(terms,0.99)
unigrams<-as.data.frame(as.matrix(sparse))
colnames(unigrams)<-make.names(colnames(unigrams))

#bigrams
bigrams<-function(x){unlist(lapply(ngrams(words(x),2), paste,collapse=" "),use.names=FALSE)}
terms<-DocumentTermMatrix(train_corpus,control=list(tokenize=bigrams,weighting=tfidf))
sparse<-removeSparseTerms(terms,0.99)
bigrams<-as.data.frame(as.matrix(sparse))
colnames(bigrams)<-make.names(colnames(bigrams))

train<-cbind(train,unigrams,bigrams)
train<-train[,-1]

#test
#unigrams
tfidf<-function(x){weightTfIdf(x,normalize=FALSE)}
terms<-DocumentTermMatrix(test_corpus,control=list(weighting=tfidf))
unigrams<-as.data.frame(as.matrix(terms))
colnames(unigrams)<-make.names(colnames(unigrams))

#bigrams
bigrams<-function(x){unlist(lapply(ngrams(words(x),2), paste,collapse=" "),use.names=FALSE)}
terms<-DocumentTermMatrix(test_corpus,control=list(tokenize=bigrams,weighting=tfidf))
bigrams<-as.data.frame(as.matrix(terms))
colnames(bigrams)<-make.names(colnames(bigrams))

test<-cbind(test,unigrams,bigrams)
test<-test[,-1]
rm(neg,pos,add,contractions,sparse,terms)

df<-data.frame(matrix(ncol=length(names(train)[names(train)!=names(test)]),nrow=nrow(test)))
colnames(df)<-names(train)[names(train)!=names(test)]
df[is.na(df)]<-0
test_matched<-cbind(test[,names(test)[names(train)==names(test)]],df)

#logit
logit_fit<-train(sentiment~.,data=train,
                 method="glm",
                 trControl=trainControl(method="cv",number=5,classProbs=TRUE,verboseIter=TRUE))
sent_predict_logit<-predict(logit_fit, newdata=test_matched)
eval_logit<-confusionMatrix(test$sentiment,sent_predict_logit)

#random forest
rf_fit<-train(sentiment~.,data=train,
              method="ranger",
              trControl=trainControl(method="cv",number=5,classProbs=TRUE,verboseIter=TRUE),
              tuneGrid=expand.grid(mtry=41,splitrule="gini",min.node.size=1))
sent_predict_rf<-predict(rf_fit, newdata=test_matched)
eval_rf<-confusionMatrix(test$sentiment,sent_predict_rf)

#gradient boosting
gbm_fit<-train(sentiment~.,data=train,
               method="gbm",
               trControl=trainControl(method="cv",number=5,classProbs=TRUE,verboseIter=TRUE),
               tuneGrid=expand.grid(n.trees=c(100,300),interaction.depth=c(1,3),
                                    shrinkage=0.1,n.minobsinnode=c(3,6)))
sent_predict_gbm<-predict(gbm_fit, newdata=test_matched)
eval_gbm<-confusionMatrix(test$sentiment,sent_predict_gbm)

#nn
nn_fit<-train(sentiment~.,data=train,
                 method="avNNet",
                 trControl=trainControl(method="cv",number=5,classProbs=TRUE,verboseIter=TRUE))
sent_predict_nn<-predict(nn_fit, newdata=test_matched)
eval_nn<-confusionMatrix(test$sentiment,sent_predict_nn)

